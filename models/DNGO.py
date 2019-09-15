from robo.models.base_model import BaseModel
from robo.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_unnormalization

import numpy as np
import torch
from torch.autograd import Variable
import sys
sys.path.append('/home/rkohli/aml_projects/src/config_generators/')


from .SimpleNeuralNet import Net
from scipy import optimize

class DG(BaseModel):
    
    def __init__(self, num_epochs=20000,
                 learning_rate=0.01, momentum=0.9,
                 adapt_epoch=5000, prior=None,
                 H=50,
                 D=10, alpha=1.0, beta=1000):
        """
        A pytorch implementation of Deep Networks for Global Optimizatin [1]. This module performas Bayesian Linear Regression with basis function extracted from a
        neural network.
        
        [1] J. Snoek, O. Rippel, K. Swersky, R. Kiros, N. Satish, 
            N. Sundaram, M.~M.~A. Patwary, Prabhat, R.~P. Adams
            Scalable Bayesian Optimization Using Deep Neural Networks
            Proc. of ICML'15
            
        Parameters
        ----------

            
        """
        self.X = None
        self.y = None
        self.network = None
        self.alpha = alpha
        self.beta = beta
        self.init_learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.H = H # the neural number of the middle layers
        self.D = D # size of the last hidden layer
        
    def train(self, X, Y, do_optimize=False):
        """
        Trains the model on the provided data.
        The training data base can be enriched.
        Parameters
        ----------
        X: np.ndarray (N, D)
            Input datapoints. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        self.X: torch float tensor of the normalized input(X)
        Y: np.ndarray (N, T)
            The corresponding target values.
            The dimensionality of Y is (N, T), where N has to
            match the number of points of X and T is the number of objectives
        self.Y: torch float tensor of the normalized Y
        """
        # Normalize inputs        
        (normX, normY) = self.normalize(X, Y)
        self.X = Variable(torch.from_numpy(normX).float())
        self.y = Variable(torch.from_numpy(normY).float(), requires_grad=False)
        features = X.shape[1]
        self.network = Net(features, self.H, self.D, 1) # here we suppose that D_out = 1
        loss_fn = torch.nn.MSELoss(size_average=True)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.init_learning_rate)
        for t in range(self.num_epochs):
            y_pred = self.network(self.X)
            loss = loss_fn(y_pred, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.phi = self.network.PHI(self.X).data
        res = optimize.fmin(self.marginal_log_likelihood, np.random.rand(2))
        self.hypers = [np.exp(res[0]), np.exp(res[1])]
        return(self.hypers)
        
        
    def marginal_log_likelihood(self, theta): # theta are the hyperparameters to be optimized

        if np.any([t>10 or t<-5 for t in theta]):#(-5 > theta) + (theta > 10)):
            return -1e25
        alpha = np.exp(theta[0]) # it is not clear why here we calculate the exponential
        beta = np.exp(theta[1])
        Ydata = self.y.data # for the bayesian part, we do not need Y to be a variable anymore
        D = self.X.size()[1]
        N = self.X.size()[0]
        Identity = torch.eye(self.phi.size()[1])
        self.phi_T = torch.transpose(self.phi, 0, 1)
        self.K = torch.addmm(beta, Identity, alpha, self.phi_T, self.phi)
        self.K_inverse = torch.inverse(self.K)
        m = beta*torch.mm(self.K_inverse, self.phi_T)
        self.m = torch.mv(m, Ydata)
        mll = (D/2.)*np.log(alpha)
        mll += (N/2.)*np.log(beta)
        mll -= (N/2.) * np.log(2*np.pi)
        mll -= (beta/2.)* torch.norm(Ydata - torch.mv(self.phi, self.m),2)
        mll -= (alpha/2.) * torch.dot(self.m,self.m)
        Knumpy = self.K.numpy() # convert K to numpy for determinant calculation
        mll -= 0.5*np.log(np.linalg.det(Knumpy))
        return mll
    
    def negative_mll(self, theta):
        """
        Returns the negative marginal log likelihood (for optimizing it with scipy).

        Parameters
        ----------
        theta: np.array(2,)
            The hyperparameter alpha and beta on a log scale

        Returns
        -------
        float
            negative lnlikelihood + prior
        """
        nll = -self.marginal_log_likelihood(theta)
        return nll

    def predict(self, xtest):
        mx = Variable(torch.from_numpy(np.array(self._mx)).float())
        sx = Variable(torch.from_numpy(np.array(self._sx)).float())
        xtest = Variable(torch.from_numpy(np.array(xtest)).float())
        xtest = (xtest - mx)/sx
        phi_test = self.network.PHI(xtest).data
        phi_T = torch.transpose(phi_test, 0, 1)
        self.marginal_log_likelihood(self.hypers)
        mean = np.dot(phi_test.numpy(), self.m)
        mean = mean*self._sy+self._my
        var = np.diag(np.dot(phi_test.numpy(),np.dot(self.K_inverse.numpy(), phi_T.numpy())))+(1./self.hypers[1])
        v = var
        v *=(self._sy**2)

        return mean, var
    
    def normalize(self, x, y):
        col=x.shape[1]
        row=x.shape[0]
        mx=list()
        sx=list()
        for i in range(col):
            mx.append(np.mean(x[:,i]))
            sx.append(np.std(x[:,i],ddof=1))
        my=np.mean(y)
        sy=np.std(y,ddof=1)
        self._mx=mx
        self._sx=sx
        self._my=my
        self._sy=sy
        mx_mat=np.mat(np.zeros((row,col)))
        sx_mat=np.mat(np.zeros((row,col)))
        for i in range(row):
            mx_mat[i,:]=mx
            sx_mat[i,:]=sx
        x_nom=(x-mx_mat)/sx_mat
        y_nom=(y-self._my)/self._sy
        return x_nom,y_nom
        
    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """

        inc, inc_value = super(DG, self).get_incumbent()

        return inc.numpy(), inc_value.numpy()