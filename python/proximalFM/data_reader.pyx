import pandas as pd
import numpy as np
cimport numpy as np
import scipy.sparse as sps
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
from cpython.string cimport PyString_AsString

# TODO add in get_features method from DataFileReader in notebook to allow for more 
# complex feature types

cdef char ** to_cstring_array(list_str):
    cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
        ret[i] = PyString_AsString(list_str[i])
    return ret

cdef class DataReader(object): 
    cdef unsigned int _label_idx;
    cdef unsigned int[:] _features_idx;
    cdef char ** _features;
    cdef unsigned int _features_dim;
    cdef unsigned int _D
    
    def __cinit__(self, object data_dictionary):
        ''' Instantiate a DataReader object for ingesting a pandas data-frame from
            file or memory, and convert to a sparse matrix of hashed features using one-hot-
            encoding 
        '''
        cdef unsigned int n_features
        cdef unsigned int i
        self._label_idx = data_dictionary['header'].index(data_dictionary['label'])
        n_features = len(data_dictionary['features'])
        self._features_idx = np.zeros(n_features, dtype = np.uint32)
        for i in range(n_features):
            self._features_idx[i] = data_dictionary['header'].index(data_dictionary['features'][i])  
        self._features = to_cstring_array(data_dictionary['features'])
        self._features_dim = data_dictionary['features_dim']
        self._D = 2**self._features_dim
        
    def __dealloc__(self):
        ''' Ensure dynamically allocated objects have their memory freed
        '''
        free(self._features)
        
    def load_data(self, path, test_data=False):
        """ Load data from a csv file as a pandas data frame and convert to a numpy 
            array. 
            Args:
                path (str): A path to the CSV file containing the raw data
            Returns:
                X (numpy.array): A matrix of num_observations rows, num_features cols 
                containing observed categorical values for columns selected as "features"
                y (numpy.array): A matrix of num_observations rows, 1 col containing 
                named target variable values (0 or 1)
        """
        # TODO need to add in error checking in case data does not conform to data dictionary
        X = pd.read_csv(path).as_matrix()
        if test_data:
            return X
        y = X[:,self._label_idx]
        X = X[:,self._features_idx]
        return X,y
                  
    def transform(self, X):
        """ Encode categorical columns into sparse matrix with one-hot-encoding
            Args:
                X (numpy.array): matrix of categorical data to encode
                S (scipy.sparse.csr_matrix) sparse matrix encoding categorical variables 
                into sparse feature indices
        """
        # TODO add in get_features calls and methods 
        cdef long int[:] rows, cols
        cdef long int[:] data
        cdef long int nnz

        nnz = np.count_nonzero(X)
        rows = np.zeros(nnz, dtype=np.int)
        cols = np.zeros(nnz, dtype=np.int)
        data = np.zeros(nnz, dtype=np.int)
        idx = 0
        nrows = X.shape[0]
    
        for row in range(0,nrows):
            for col in np.nonzero(X[row,])[0]:
                cols[idx] = np.int(abs(hash('{}_{}'.format(self._features[col], X[row,col]))) % self._D)
                rows[idx] = np.int(row)
                data[idx] = 1
                idx = idx+1
        ncols = 1+np.max(cols)

        X_new = sps.coo_matrix((data,(rows,cols)),shape=(nrows,ncols))
        X_new = X_new.tocsr()
        return X_new
