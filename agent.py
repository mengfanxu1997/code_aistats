import torch, cvxpy
import numpy as np
import networkx as nx
col_softmax = torch.nn.Softmax(dim=1)

class FedExp3:
    def __init__(self, n_agents, n_arms, W, lr, expr_scheduler, device) -> None:
        self.Z = torch.zeros([n_agents, n_arms], device=device)
        self.W = W
        self.lr = lr
        self.expr_scheduler = expr_scheduler
        self.X0 = torch.ones([n_agents, n_arms], device=device) / n_arms
        self.device = device

    def action(self, rng):
        K = self.Z.shape[-1]
        X = col_softmax(-self.lr*self.Z)
        gamma = next(self.expr_scheduler)
        P = (1 - gamma) * X + gamma * self.X0
        A = torch.multinomial(P, num_samples=1, generator=rng)
        A_one_hot = torch.nn.functional.one_hot(A, num_classes=K).squeeze(1)
        return A_one_hot, P

    def update(self, loss_matrix, actions, probs):
        L_t = loss_matrix.to(self.device)
        G = L_t * actions / probs
        self.Z = torch.mm(self.W.float(), self.Z.float()) + G

class CommNet:
    def __init__(self, nx_graph) -> None:
        self.comm_net = nx_graph
    
    def max_deg_gossip(self, spectral_gap=False):
        degrees = [val for (node, val) in self.comm_net.degree()]
        max_deg = max(degrees)
        D = np.diag(degrees)
        A = nx.to_numpy_array(self.comm_net)
        P = np.eye(len(degrees)) - (D - A) / (max_deg+1)
        # spectral gap
        if spectral_gap:
            return P, compute_spectral_gap(P)
        return P

    def fast_gossip(self, algo, spectral_gap=False):
        if algo == 'SDP':
            comple_graph = nx.complement(self.comm_net)
            n = self.comm_net.number_of_nodes()
            P = cvxpy.Variable((n, n))
            e = np.ones(n)
            obj = cvxpy.Minimize(cvxpy.norm(P - 1.0/n))
            cnsts = [
                P@e==e,
                P.T == P,
                P >= 0
            ]
            for u, v in comple_graph.edges():
                if u != v: cnsts.append(P[u, v] == 0)
            prob = cvxpy.Problem(obj, cnsts)
            prob.solve()
            # spectral gap
            if spectral_gap:
                return P.value, compute_spectral_gap(P.value)
            return P.value
        else:
            raise NotImplementedError("The "+algo+" method has not been implemented.")



def cube_root_scheduler(gamma=0.01):
    '''Generates a series of exploration ratios'''
    step = 1
    while True:
        yield gamma / step ** (1/3) 
        step += 1

def compute_spectral_gap(P):
    singular_values = np.linalg.svd(P, compute_uv=False, hermitian=True)
    gap = 1 - singular_values[1]
    return gap

def fedexp3_ub_exact(n_epochs, n_agents, n_arms, spectral_gap, lr_array, gamma_array):
    C_w = 3 + min(
        2 * np.log(n_epochs) + np.log(n_agents),
        np.sqrt(n_agents)
    ) / spectral_gap
    lr_last = lr_array[-1]
    gamma_last = gamma_array[-1]
    cum_reg = np.log(n_arms) / lr_last
    for lr, gamma in zip(lr_array, gamma_array):
        first = n_arms**2/2 * lr / gamma
        second = n_arms**2 / gamma_last * C_w * lr
        third = gamma
        ins = first + second + third
        cum_reg += ins
        yield cum_reg

if __name__ == "__main__":
    # no communication
    N = 4
    adj = np.zeros([N, N])
    g = CommNet(
        nx.from_numpy_array(adj)
    )
    print(g.max_deg_gossip())

    # complete graph
    adj = np.ones([N, N])
    for i in range(N):
        adj[i][i] = 0
    g = CommNet(
        nx.from_numpy_array(adj)
    )
    print(g.max_deg_gossip())
    
    # sqrt(N)-by-sqrt(N) grid
    g = CommNet(nx.grid_graph([
        int(np.sqrt(N)),
        int(np.sqrt(N))
    ]))
    print(g.max_deg_gossip())
    
    # random geometric graphs with connectivity radius
    # r = 2 sqrt(log(n)^2 / n)
    r = np.sqrt(np.log(N) ** 1.1 / N)
    g = CommNet(nx.random_geometric_graph(
        N, r
    ))
    print(g.max_deg_gossip())

    