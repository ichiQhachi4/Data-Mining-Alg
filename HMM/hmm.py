import numpy as np


class HMM:
    def __init__(self, N, M, pi, A, B):
        self.N = N              # states number
        self.M = M              # observation number
        self.A = np.array(A)
        self.B = np.array(B)
        self.pi = np.array(pi)
        if self.A.shape != (N, N):
            print("Warning! State transition probabilities are not compatible with state number")
            print("shape of matrix: " + str(A.shape))
        if self.B.shape != (N, M):
            print("Warning! Output probabilities are not compatible with state number")
            print("shape of matrix: " + str(B.shape))
        if self.pi.shape != (N,):
            print("Warning! Output probabilities are not compatible with state number")
            print("shape of matrix: " + str(B.shape))

    def forward_var(self, alpha, observation_seq, t):
        if t == len(observation_seq):
            return
        alpha_t = []
        for j in range(self.N):
            tmp = 0
            for i in range(self.N):
                tmp += alpha[t-1][i] * self.A[i][j]
            alpha_t.append(tmp * self.B[j][int(observation_seq[t])])
        alpha.append(alpha_t)
        self.forward_var(alpha, observation_seq, t + 1)

    def cal_alpha(self, observation_seq):
        alpha = []
        alpha_0 = []
        for i in range(self.N):
            alpha_0.append(self.B[i][int(observation_seq[0])] * self.pi[i])
        alpha.append(alpha_0)
        t = 1
        self.forward_var(alpha, observation_seq, t)
        return alpha

    def coherence(self, observation_seq):
        # forward alg
        alpha = self.cal_alpha(observation_seq)
        return sum(alpha[len(observation_seq) - 1])

    def viterbi_var(self, sigma, phi, observation_seq, t):
        if t == len(observation_seq):
            return
        sigma_i = []
        phi_i = []
        for j in range(self.N):
            maximum = 0
            max_index = 0
            for i in range(self.N):
                tmp = sigma[t - 1][i] * self.A[i][j]
                if tmp > maximum:
                    maximum = tmp
                    max_index = i
            sigma_i.append(maximum * self.B[observation_seq[t]][j])
            phi_i.append(max_index)
        sigma.append(sigma_i)
        phi.append(phi_i)
        self.viterbi_var(sigma, phi, observation_seq, t + 1)

    def viterbi(self, observation_seq):
        sigma = []
        sigma_0 = []
        phi = []
        phi_0 = []
        for i in range(self.N):
            sigma_0.append(self.B[i][int(observation_seq[0])] * self.pi[i])
            phi_0.append(0)
        sigma.append(sigma_0)
        phi.append(phi_0)
        t = 1
        self.viterbi_var(sigma, phi, observation_seq, t)
        return sigma, phi

    def find_state_seq(self, observation_seq):
        sigma, phi = self.viterbi(observation_seq)
        q = []
        index = sigma[len(observation_seq)-1].index(max(sigma[len(observation_seq)-1]))
        q.append(index)
        for i in range(len(observation_seq) - 1):
            q.append(phi[len(observation_seq)-i - 1][q[i]])
        q.reverse()
        return q


hmm = HMM(3, 3, [30/47, 9/47, 8/47], [[0.8,0.1,0.1],[0.3,0.4,0.3],[0.4,0.2,0.4]],[[0.8,0.2,0.1],[0.1,0.5,0.2],[0.1,0.3,0.7]])
print(hmm.coherence([0,0,1,2,1,2,1,0]))
print(hmm.find_state_seq([0,0,1,2,1,2,1,0]))