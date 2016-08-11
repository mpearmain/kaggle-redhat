cimport numpy as np
import numpy as np
import random
cimport cython
from scipy.sparse import csr_matrix
from libc.math cimport exp, sqrt, log
from libc.string cimport memcpy
import pickle
import logging

cdef double logist(double x, double lim = 35.0):
    """ Bounded logistic function
    """
    return 1. / (1 + exp(-max(min(x, lim), -lim)))

cpdef logloss(double p, double y):
    """ Bounded logloss function
        Args:
            p (double): prediction
            y (double): real answer

        Returns:
            l (double) : logarithmic loss of p given y
    """ 
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

# TODO 
# i test and check results
# ii optimize cython code and add threading

cdef class ProximalFM(object):
    cdef double _alpha
    cdef double _beta
    cdef double _L1
    cdef double _L2
    cdef double _alpha_fm
    cdef double _beta_fm
    cdef double _L1_fm
    cdef double _L2_fm
    cdef unsigned int _D
    cdef unsigned int _epoch
    cdef unsigned int _report_frequency
    cdef unsigned int _fm_dim
    cdef double _fm_initDev
    cdef double[:] _n
    cdef double[:] _z
    cdef double[:] _w
    cdef unsigned int[:] _fm_initialised
    cdef double[:] _n_fm
    cdef double[:] _z_fm
    cdef double[:] _w_fm
    cdef bint _enable_fm
    cdef unsigned int _seed
    cdef bint _warm_start
    cdef bint _fitted

    def __init__(self, object model, seed=0):
        """ Instantiate a ProximalFM learner
            Args:
                model (python dictionary): a dictionary of model hyper-parameters
                seed (unsigned int): the seed for the random number generator 
        """
        self._alpha = model['alpha']
        self._beta = model['beta']
        self._L1 = model['L1']
        self._L2 = model['L2']
        self._D = model['D']
        self._epoch = model['epoch']
        self._warm_start = model['warm_start']
        self._enable_fm = model['enable_fm']
        self._seed = seed
        self._report_frequency = model['report_frequency']
        if(self._enable_fm):
            self._alpha_fm = model['alpha_fm']
            self._beta_fm = model['beta_fm']
            self._L1_fm = model['L1_fm']
            self._L2_fm = model['L2_fm']
            self._fm_dim = model['fm_dim']
            self._fm_initDev = model['fm_initDev']
        self._init_model()
            
    def _init_model(self):
        """ Initialise model weights
        """  
        self._n = np.zeros(self._D, dtype = np.float64)
        self._z = np.zeros(self._D, dtype = np.float64)
        self._w = np.zeros(self._D, dtype = np.float64)       
        random.seed(self._seed)
        if(self._enable_fm):
            self._n_fm = np.zeros(self._D * self._fm_dim, np.float64)
            self._z_fm = np.zeros(self._D * self._fm_dim, np.float64)
            self._w_fm = np.zeros(self._D * self._fm_dim, np.float64)
            self._fm_initialised = np.zeros(self._D, np.uint32) 
        self._fitted = False
        
    def __getstate__(self):
        ''' Returns the state of the instance and is pickled instead of the contents
            of the instance's dict. This is in order to support pickling of cdef classes
            Returns:
                state (tuple): state of the instance
        '''
        nz = np.nonzero(np.asarray(self._n))
        state = [ self._alpha, 
                    self._beta, 
                    self._L1, 
                    self._L2,
                    self._D,         
                    pickle.dumps(np.asarray(nz)),
                    pickle.dumps(np.asarray(self._n)[nz]),
                    pickle.dumps(np.asarray(self._z)[nz]),
                    pickle.dumps(np.asarray(self._w)[nz]),
                    self._epoch, 
                    self._report_frequency,
                    self._fm_dim,
                    self._fm_initDev,
                    self._enable_fm,
                    self._seed,
                    self._warm_start,
                    self._fitted
                ]
        if(self._enable_fm):
            nz_fm = np.nonzero(np.asarray(self._n_fm))
            state.extend([ 
                    self._alpha_fm,
                    self._beta_fm,
                    self._L1_fm,
                    self._L2_fm,
                    pickle.dumps(np.asarray(self._fm_initialised)),
                    pickle.dumps(np.asarray(nz_fm)),
                    pickle.dumps(np.asarray(self._n_fm)[nz_fm]),
                    pickle.dumps(np.asarray(self._z_fm)[nz_fm]),
                    pickle.dumps(np.asarray(self._w_fm)[nz_fm])
                ])
        return state
           
    def __setstate__(self, state):
        ''' Receives the state object and applies it to the instance
            Args:
                state(tuple): pickled objects for setting on the instance
        '''
        self._alpha = state[0]
        self._beta = state[1]
        self._L1 = state[2]
        self._L2 = state[3]
        self._D = state[4]
        nz = np.asarray(pickle.loads(state[5]))
        self._n = np.zeros(self._D, np.float64)
        np.asarray(self._n)[nz] = pickle.loads(state[6])
        self._z = np.zeros(self._D, np.float64)
        np.asarray(self._z)[nz] = pickle.loads(state[7])
        self._w = np.zeros(self._D, np.float64)
        np.asarray(self._w)[nz] = pickle.loads(state[8])
        self._epoch = state[9]
        self._report_frequency = state[10]
        self._fm_dim = state[11]
        self._fm_initDev = state[12]
        self._enable_fm = state[13]
        self._seed = state[14]
        self._warm_start = state[15]
        self._fitted = state[16]
        if(self._enable_fm):
            self._alpha_fm = state[17]
            self._beta_fm = state[18]
            self._L1_fm = state[19]
            self._L2_fm = state[20]
            self._fm_initialised = pickle.loads(state[21])
            nz_fm = np.asarray(pickle.loads(state[22]))
            self._n_fm = np.zeros(self._D * self._fm_dim, np.float64)
            np.asarray(self._n_fm)[nz_fm] = pickle.loads(state[23])
            self._z_fm = np.zeros(self._D * self._fm_dim, np.float64)
            np.asarray(self._z_fm)[nz_fm] = pickle.loads(state[24])
            self._w_fm = np.zeros(self._D * self._fm_dim, np.float64)
            np.asarray(self._w_fm)[nz_fm] = pickle.loads(state[25])  
        
    @cython.cdivision(True)
    def _non_linear_predict_one(self, int[:] x, double w0 = 0.):
        """ Calculate the prediction contribution from the non-linear feature interactions
            Args:
                x (list of int): a list of hash indices of non-zero features
                w0 (double): initial sum of weights
            Returns:
                wTx (double): sum of non-linear weight interactions
        """
        cdef int len_x
        cdef int i
        cdef int k
        cdef double wx
        cdef double vx
        cdef double vx2
        cdef int x_i
        cdef int nk
        cdef double wTx
        cdef double Vx 
        cdef double alpha_fm
        cdef double beta_fm
        cdef double L1_fm
        cdef double L2_fm
        cdef double sign
        cdef double z_fm_i
        cdef double n_fm_i
        cdef double w_fm_i
        cdef int idx0
        cdef int idx
 
        wTx = w0
        if not self._enable_fm:
            return wTx
        
        len_x = x.shape[0]
        nk = self._fm_dim
        alpha_fm = self._alpha_fm
        beta_fm = self._beta_fm
        L1_fm = self._L1_fm
        L2_fm = self._L2_fm
        
        Vx = 0.
        for i in range(len_x):
            x_i = x[i]
            if not self._fm_initialised[x_i]:
                idx0 = x_i * nk 
                for k in range(nk):
                    idx = idx0 + k
                    z_fm_i = self._z_fm[idx]
                    w_fm_i = self._w_fm[idx]
                    z_fm_i = random.gauss(0., self._fm_initDev)
                    sign = -1. if z_fm_i < 0. else 1.                   
                    if sign * z_fm_i <= L1_fm:
                        w_fm_i = 0.
                    else:
                        w_fm_i = (sign * L1_fm - z_fm_i) / \
                        (beta_fm / alpha_fm + L2_fm)
                    self._z_fm[idx] = z_fm_i
                    self._w_fm[idx] = w_fm_i
                self._fm_initialised[x_i] = 1
                
        for k in range(nk):
            vx = 0.
            vx2 = 0.
            for i in range(len_x):
                x_i = x[i]
                wx = self._w_fm[x_i * nk + k]
                vx += wx
                vx2 += (wx * wx)
            vx *= vx
            Vx += 0.5*(vx - vx2)
        wTx += Vx
        
        return wTx
           
    @cython.cdivision(True)
    def _linear_predict_one(self, int[:] x, double w0 = 0.):
        """ Calculate the prediction contribution from the linear features
            Args:
                x (list of int): a list of hash indices of non-zero features
                w0 (double): initial sum of weights
            Returns:
                wTx (double): sum of linear weights
        """
        cdef int len_x
        cdef int i
        cdef int x_i
        cdef double w_i
        cdef double wTx
  
        len_x = x.shape[0]
        wTx = w0
        for i in range(len_x):
            x_i = x[i]
            w_i = self._w[x_i]
            wTx += w_i
            
        return wTx
    
    @cython.cdivision(True)
    def predict_one_with_products(self, int[:] x, int[:] product_indptr, int[:] product_indices):
        """ Predict for the set of input features combined with each of the products
            Args:
                x (list of int): a list of hash indices of non-zero input features
                product_indptr (list of int): a list of product index references
                product_indices (list of int): a list of hash indices for each 
                corresponding product 
            Returns:
                p (numpy.array): a probability prediction for input features combined
                with each product
        """
        cdef double wTx
        cdef double wTxp
        cdef int[:] p
        cdef int ip
        
        nproducts = product_indptr.shape[0] - 1
        if nproducts < 1:
            return self.predict_one(x)
 
        wTx = self._w[0]
        wTx = self._linear_predict_one(x, wTx)
 
        prob = np.zeros((nproducts, 1), dtype = np.float64)
        for ip in range(nproducts):
            p = product_indices[product_indptr[ip]:product_indptr[ip + 1]]
            wTxp = self._linear_predict_one(p, wTx)
            if self._enable_fm:
                wTxp = self._non_linear_predict_one(np.concatenate((x, p)), wTxp)
            prob[ip, 0] = logist(wTxp)
            
        return np.asarray(prob)
    
    @cython.cdivision(True)
    def predict_one(self, int[:] x):
        """ Predict for features
            Args:
                x (list of int): a list of hash indices of non-zero features
            Returns:
                p (double): a probability prediction for input features
        """
        cdef double wTx
        
        wTx = self._w[0]
        wTx = self._linear_predict_one(x, wTx)
        
        if self._enable_fm:
            wTx = self._non_linear_predict_one(x, wTx)
           
        return logist(wTx)
   
    def predict_proba_with_products(self, X, products):
        """ Predictictions for a sparse matrix X and optional sparse matrix of products
            Args:
                X (scipy.sparse.csr_matrix): a sparse matrix for input features
                products (scipy.sparse.csr_matrix): a sparse matrix for product features
            Returns:
                prob (numpy.array): probability predictions for input and product features
        """
        cdef int nrows
        cdef int[:,:] y_hat
        cdef int row
        cdef int ip
        cdef int num_products
        
        nrows = X.shape[0]
        num_products = products.shape[0] 
        prob = np.zeros((nrows * num_products, 1), dtype = np.float64)
        for row in range(nrows):
            rows = np.asarray(range(num_products)) + row * num_products
            prob[rows,] = self.predict_one_with_products(X[row].indices, products.indptr, products.indices)
        return prob
    
    def predict_proba(self, X, products = None):
        """ Predictictions for a sparse matrix X and optional sparse matrix of products
            Args:
                X (scipy.sparse.csr_matrix): a sparse matrix for input features
                products (scipy.sparse.csr_matrix): a sparse matrix for product features (optional)
            Returns:
                prob (numpy.array): probability predictions for input (and optional product) features
        """
        cdef int nrows
        cdef int[:,:] y_hat
        cdef int row
        cdef int ip
     
        if not products is None:
            return self.predict_proba_with_products(X, products)
       
        nrows = X.shape[0]
        prob = np.zeros((nrows, 1), dtype = np.float64)
        for row in range(nrows):
            prob[row, 0] = self.predict_one(X[row].indices)
        return prob
    
    def predict(self, X, products = None):
        """ Predictictions for a sparse matrix X
            Args:
                X (scipy.sparse.csr_matrix): a sparse matrix for input features
                products (scipy.sparse.csr_matrix): a sparse matrix for product features (optional)
            Returns:
                y_hat (numpy.array): probability [0,1] for input (and optional product) features
        """
        cdef int nrows
        cdef int[:,:] y_hat
        cdef double[:,:] prob
       
        prob = self.predict_proba(X, products)
        nrows = prob.shape[0]
        y_hat = np.zeros((nrows, 1), dtype = np.int32)
        for row in range(nrows):
            y_hat[row,0] = 1 if prob[row,0] >= 0.5 else 0
    
        return y_hat
        
    @cython.cdivision(True)
    def fit_one_fm(self, int[:] x, double p, double y):
        """ Update the FM model weights
            Args:
                x (list of int): a list of hash indices of non-zero features
                p (double): probability prediction of our model
                y (double): target
            Updates:
                self._n_fm (numpy array): increase by squared gradient
                self._z_fm (numpy array): weights
                self._w_fm (numpy array): weights
        """
        cdef double g
        cdef double g_fm
        cdef double g2_fm
        cdef double[:] fm_sum
        cdef double sigma
        cdef double sign
        cdef double z_fm_i
        cdef double n_fm_i
        cdef double w_fm_i
        cdef double sum_k
        cdef int len_x 
        cdef int i
        cdef int k
        cdef int x_i
        cdef int idx
        cdef unsigned int[:] missing
        cdef int nk
        cdef double alpha_fm
        cdef double beta_fm
        cdef double L1_fm
        cdef double L2_fm
        
        g = p - y 
        len_x = x.shape[0]
        nk = self._fm_dim
        alpha_fm = self._alpha_fm
        beta_fm = self._beta_fm
        L1_fm = self._L1_fm
        L2_fm = self._L2_fm
        # sum the gradients for the FM interaction weights
        fm_sum = np.zeros(nk, dtype = np.float64)
        for i in range(len_x):
            x_i = x[i]
            idx = x_i * nk
            for k in range(nk):
                fm_sum[k] += self._w_fm[idx + k]
                
        for i in range(len_x):
            x_i = x[i]
            for k in range(nk):
                idx = x_i * nk + k
                w_fm_i = self._w_fm[idx]
                n_fm_i = self._n_fm[idx]
                z_fm_i = self._z_fm[idx]
                g_fm = g * (fm_sum[k] - w_fm_i)
                g2_fm = g_fm * g_fm
                sigma = (sqrt(n_fm_i + g2_fm) - sqrt(n_fm_i)) / alpha_fm
                z_fm_i += g_fm - sigma * w_fm_i 
                n_fm_i += g2_fm
                sign = -1. if z_fm_i < 0. else 1.                   
                if sign * z_fm_i <= L1_fm:
                    w_fm_i = 0.
                else:
                    w_fm_i = (sign * L1_fm - z_fm_i) / \
                    ((beta_fm + sqrt(n_fm_i)) / alpha_fm + L2_fm)
                self._z_fm[idx] = z_fm_i
                self._n_fm[idx] = n_fm_i
                self._w_fm[idx] = w_fm_i
                                                    
    @cython.cdivision(True)
    cdef fit_one(self, int[:] x, double p, double y):
        """ Update the model
            Args:
                x (list of int): a list of hash indices of non-zero features
                p (double): probability prediction of our model
                y (double): target
            Updates:
                self._n (numpy.array): increase by squared gradient
                self._z (numpy.array): weights
                self._w (numpy.array): weights
        """
        cdef double g
        cdef double g2
        cdef double sigma
        cdef double sign
        cdef double z_i
        cdef double n_i
        cdef double w_i
        cdef double L1
        cdef double L2
        cdef double alpha
        cdef double beta
        cdef int x_i
        cdef int len_x
        cdef int i
        g = p - y # gradient under logloss
        g2 = g * g
        L1 = self._L1
        L2 = self._L2
        alpha = self._alpha
        beta = self._beta
        
        len_x = 1 + x.shape[0]
        # update z and n
        for i in range(len_x):
            x_i = 0 if i == 0 else x[i-1]
            z_i = self._z[x_i]
            n_i = self._n[x_i]
            w_i = self._w[x_i]
            sigma = (sqrt(n_i + g2) - sqrt(n_i)) / alpha
            z_i += g - sigma * w_i
            n_i += g2
            sign = -1. if z_i < 0 else 1. 
            if sign * z_i <= L1:
                w_i = 0.
            else:
                w_i = (sign * L1 - z_i) / \
                    ((beta + sqrt(n_i)) / alpha + L2)
            self._z[x_i] = z_i
            self._n[x_i] = n_i
            self._w[x_i] = w_i
 
        if self._enable_fm:
            self.fit_one_fm(x, p, y)
    
    def fit(self, X, y_in):
        """ Update the model with a sparse input feature matrix and its targets
            Args:
                X (scipy.sparse.csr_matrix): a sparse matrix of hash indices of non-zero features
                y_in (numpy.array): targets
            Returns:
            updated model weights
        """
        cdef double progressiveLoss
        cdef int progressiveCount
        cdef double p
        cdef double target
        cdef double loss
        cdef int epoch
        cdef int row
        cdef int[:] x
        cdef double L1_fm
        cdef double L2_fm 
        if not (self._warm_start and self._fitted):
            self._init_model()
        L1_fm = self._L1_fm
        L2_fm = self._L2_fm
        for epoch in range(self._epoch):
            progressiveLoss = 0.
            progressiveCount = 0
            # if first epoch, do not use L1_fm or L2_fm
            if epoch == 0:
                self._L1_fm = 0.
                self._L2_fm = 0.
            else:
                self._L1_fm = L1_fm
                self._L2_fm = L2_fm
            for row in range(X.shape[0]):
                x = X[row].indices
                p = self.predict_one(x)
                target = y_in[row]
                loss = logloss(p, target)
                self.fit_one(x, p, target)
                progressiveLoss += loss
                progressiveCount +=1 
                if row % self._report_frequency == 0:
                    logging.debug("Epoch: %d\tcount: %d\tLoss: %f\tProgressive Loss: %f" % \
                                  (epoch, row, loss, progressiveLoss / progressiveCount))
            logging.debug('Epoch %d finished' % epoch)
        self._fitted = True
        return self
    
    def transform(self, X, y=None):
        return X