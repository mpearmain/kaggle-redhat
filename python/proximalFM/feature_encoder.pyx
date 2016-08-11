import numpy as np
cimport numpy as np
import scipy.sparse as sps
from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString,PyString_InternFromString
import pickle
from sklearn.utils.murmurhash import murmurhash3_bytes_s32 

# TODO 
# i add in get_features method from DataFileReader in notebook to allow for more 
# complex feature types
# ii can we make use instead of scikit-learn FeatureHasher class?

cdef char ** to_cstring_array(list_str):
    cdef int i
    cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
        ret[i] = PyString_AsString(list_str[i])
    return ret

cdef class FeatureEncoder(object): 
    cdef unsigned int _label_idx;
    cdef unsigned int[:] _features_idx;
    cdef char ** _features;
    cdef unsigned int _features_dim;
    cdef unsigned int _D
    
    def __init__(self, object data_dictionary):
        ''' Instantiate a FeaturesEncoder object for transforming a pandas data-frame from
            to a sparse matrix of hashed features using one-hot-encoding 
        '''
        cdef unsigned int n_features
        cdef unsigned int i
        self._label_idx = -1
        if 'label' in data_dictionary:
            self._label_idx = int(data_dictionary['header'].index(data_dictionary['label'])) 
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
        
    def __getstate__(self):
        state = []
        state.append(self._label_idx)
        state.append(pickle.dumps(np.asarray(self._features_idx)))
        state.append(self._features_dim)
        state.append(self._D)
        i = 0
        features = list()
        while i < self._features_dim:
            feature_name = PyString_InternFromString(self._features[i])
            features.append(feature_name)
            i += 1
        state.append(features)
        return state

    def __setstate__(self, state):
        self._label_idx = state[0]
        #self._features_idx = np.asarray(pickle.loads(state[1])),
        _features_idx = np.asarray(pickle.loads(state[1]))
        self._features_dim = state[2]
        self._D =  state[3]
        features = state[4]
        self._features = to_cstring_array(features)
                      
    def transform(self, X_in):
        """ Encode categorical columns into sparse matrix with one-hot-encoding
            Args:
                X (numpy.ma.masked_array): matrix of categorical data to encode
                S (scipy.sparse.csr_matrix) sparse matrix encoding categorical variables 
                into sparse feature indices
        """
        # TODO add in get_features calls and methods 
        cdef int[:] rows, cols
        cdef int[:] data
        cdef int nnz
        cdef int idx
        cdef int nrows
        cdef int row
        cdef int col
        
        X = X_in[:,self._features_idx]
        
        if not isinstance(X, np.ma.masked_array):
            X = np.ma.asarray(X)
        
        nnz = X.count()
        rows = np.zeros(nnz, dtype=np.int32)
        cols = np.zeros(nnz, dtype=np.int32)
        data = np.zeros(nnz, dtype=np.int32)
        idx = 0
        nrows = X.shape[0]
        ncols = X.shape[1]
    
        for row in range(0,nrows):
            # np.ma.compressed(X[row,])
            row_mask = np.ma.getmask(X[row,])
            for col in range(0,ncols):
                if row_mask[col]:
                    continue
                h = murmurhash3_bytes_s32('{}_{}'.format(self._features[col], X[row,col]), 0)
                cols[idx] = abs(h) % self._D
                rows[idx] = row
                data[idx] = 1
                idx = idx+1
        ncols = 1+np.max(cols)

        X_out = sps.coo_matrix((data,(rows,cols)),shape=(nrows,ncols))
        X_out = X_out.tocsr()
        return X_out

