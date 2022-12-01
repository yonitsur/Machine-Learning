#################################
# Your name: Yehonatan Tsur, 204617963
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import portion

class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.sort(np.random.uniform(0, 1, (m,2)), axis=0)
        for i in range(m):
            if(X[i][0]<=0.2 or (X[i][0]<=0.6 and X[i][0]>=0.4) or X[i][0]>=0.8):
                X[i][1] = np.random.choice(2, 1, p=[0.2, 0.8])
            else:
                X[i][1] = np.random.choice(2, 1, p=[0.9, 0.1])
        return X

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        true_error = np.array([])
        empirical_error = np.array([])
        ep = np.zeros(T) # ep = true error
        es = np.zeros(T) # es = empirical error
        N = range(m_first, m_last + 1, step)

        for n in N:
            for t in range(T):
                S = self.sample_from_D(n)
                I, errors = intervals.find_best_interval(S[:,0], S[:,1], k)
                ep[t] = self.true_error(I)
                es[t] = errors/n
            true_error = np.append(true_error, ep.mean())
            empirical_error = np.append(empirical_error, es.mean())

        plt.plot(N, empirical_error, label='empirical error')
        plt.plot(N, true_error, label='true error')
        plt.xlabel("n")
        plt.ylabel("error")
        plt.legend()
        plt.show()
        return np.column_stack((empirical_error, true_error))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        K = range(k_first, k_last + 1, step)
        ep = np.array([])  # ep = true error
        es = np.array([])  # es = empirical error
        S = self.sample_from_D(m)
        X = S[:,0]
        Y = S[:,1]

        for k in K:
            I, errors = intervals.find_best_interval(X, Y, k)
            ep = np.append(ep, self.true_error(I))
            es = np.append(es, errors / m)

        plt.plot(K, es, label='empirical error')
        plt.plot(K, ep, label='true error')
        plt.xlabel("k")
        plt.ylabel("error")
        plt.legend()
        plt.show()
        return np.argmin(es)

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        K = np.arange(k_first, k_last + 1, step)
        ep = np.array([])  # ep = true error
        es = np.array([])  # es = empirical error
        S = self.sample_from_D(m)
        X = S[:,0]
        Y = S[:,1]
        penalty = 2 * np.sqrt((2 * K + np.log(20)) / m)

        for k in K:
            I, errors = intervals.find_best_interval(X, Y, k)
            ep = np.append(ep, self.true_error(I))
            es = np.append(es, errors / m)

        plt.plot(K, ep, label='true error')
        plt.plot(K, es, label='empirical error')
        plt.plot(K, penalty, label='penalty')
        plt.plot(K, es + penalty, label='empirical error + penalty')
        plt.xlabel("k")
        plt.ylabel("error")
        plt.legend()
        plt.show()
        return np.argmin(es + penalty)*step + k_first

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        best_k = 1
        min_error = 1
        S = self.sample_from_D(m)
        np.random.shuffle(S)
        train_set = np.array(sorted(S[:int(0.8*m)], key=lambda x: x[0]))
        holdout_set = S[int(0.8*m):]
        X = train_set[:, 0]
        Y = train_set[:, 1]
        Xh = holdout_set[:, 0]
        Yh = holdout_set[:, 1]

        for k in range(1, 11):
            I, errors = intervals.find_best_interval(X, Y, k)
            eh = self.empirical_error(I, Xh, Yh)
            if (eh < min_error):
                min_error = eh
                best_k = k

        return best_k

    #################################
    # Place for additional methods

    def true_error(self, I):
        """Calculates the true error of hypothesis hI.
        Input: I - a list of intervals.

        Returns: The true error (a float) = = P[hI(X)!=Y].
        """
        I=[portion.closed(I[i][0],I[i][1]) for i in range(len(I))]
        A=[portion.closed(0,0.2), portion.closed(0.4,0.6), portion.closed(0.8,1)]
        B=[portion.closed(0.2,0.4), portion.closed(0.6,0.8)]
        E_A = 0
        E_B = 0

        for interval in I:
            for a in A:
                if not (interval & a).empty:
                    E_A += (interval & a).upper-(interval & a).lower
            for b in B:
                if not (interval & b).empty:
                    E_B += (interval & b).upper-(interval & b).lower

        return (E_A*0.2)+((0.6-E_A)*0.8)+(E_B*0.9)+((0.4-E_B)*0.1)

    def empirical_error(self, I, X, Y):
        """Calculates the empirical error of hI on X,Y.
        Input: I -  a list of intervals.
               X - data sample
               Y - lables of X
        Returns: The empirical error (a float) = (1/len(X))*sum[∆zo(hI(Xi),Yi)]
        """
        I = [portion.closed(I[i][0], I[i][1]) for i in range(len(I))]
        err = 0
        for x in range(len(X)):
            Y_x = 0
            for interval in I:
                if X[x] in interval:
                    Y_x = 1
                    break
                elif X[x] < interval.lower:
                    break
            err += (Y_x != Y[x])
        return err/len(X)

    #################################

if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
    

