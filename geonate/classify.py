"""
Classification module.

"""
# import common packages 
from typing import AnyStr, Dict, Optional

##############################################################################################

##############################################################################################
##                                                                              CLUSTERING                                                                                                    #
##                                                                                                                                                                                                         #
# =========================================================================================== #
#              K-Means clustering
# =========================================================================================== #    
def kmeans(input, n_cluster=3, max_iter=500, algorithm='lloyd', **kwargs):
    """
    Perform K-Means clustering for raster image. Kmeans is a fast and simple algorithm. 

    Args:
        input (rasterio.DatasetReader or np.ndarray): Multispectral input data. Can be a raster image or a numpy array.
        n_cluster (int): Number of clusters to form. Default is 3.
        max_iter (int): Maximum number of iterations of the k-means algorithm for a single run. Default is 300.
        algorithm (str): K-means algorithm to use. The available algorithms include "lloyd" and "elkan". "elkan" variation can be more efficient on some datasets with well-defined clusters, by using the triangle inequality. However it’s more memory intensive due to the allocation of an extra array of shape. Default is 'lloyd'.
        **kwargs: Additional keyword arguments to pass to the KMeans model.

    Returns:
        np.ndarray or rasterio.DatasetReader: K-Means clustering result in the same format as the input.

    """
    import numpy as np
    import rasterio
    from sklearn.cluster import KMeans
    from .common import array2raster, reshape_raster

    # Identify datatype and define input data
    # Raster image
    if isinstance(input, rasterio.DatasetReader):
        arr = input.read()
        height, width = input.shape
        nbands =  input.count
        meta = input.meta
    # Data Array
    elif isinstance(input, np.ndarray):
        if len(input.shape) < 3:
            raise ValueError('Input must be multispectral data (multi-band)')
        else:
            arr = input
            nbands, height, width = input.shape

    else: 
        raise ValueError('Input is not supported')
    
    # Reshape from raster to image format, and from 3D to 2D
    arr_reshape_img = reshape_raster(arr, mode='image')
    print(arr_reshape_img.shape)
    img_reshaped = arr_reshape_img.reshape((-1, nbands))

    # Define KMeans model and fit the KMeans model
    kmean_model = KMeans(n_clusters= n_cluster, max_iter= max_iter, algorithm= algorithm, **kwargs)
    kmean_fit = kmean_model.fit(img_reshaped)

    # Extract labels and reshape based on input image
    labels = kmean_fit.labels_
    km_results = labels.reshape((height, width))

    # Return output based on input similar to input
    if isinstance(input, np.ndarray):
        return km_results
    
    elif isinstance(input, rasterio.DatasetReader):
        meta.update({'count': 1})
        km_results_rast = array2raster(km_results, meta)
        return km_results_rast
    

# =========================================================================================== #
#              Train, Tune, and Classify image using Random Forest
# =========================================================================================== #    
class RandomForest:
    """
    A class that encapsulates a Random Forest model for classification tasks, including model training,
    hyperparameter tuning using grid or random search, and classification of image data.

    Attributes:
        X_train (ndarray or DataFrame): The training features for model fitting.
        y_train (ndarray or Series): The training labels for model fitting.
        X_test (ndarray or DataFrame): The test features for model validation.
        y_test (ndarray or Series): The test labels for model validation.
        initial_rf (RandomForestClassifier, optional): The initial Random Forest model (untuned).
        tuned_rf (RandomForestClassifier, optional): The Random Forest model after tuning using grid or random search.
        accuracy (float, optional): Accuracy of the initial (naive) Random Forest model.
        confusion_matrix (ndarray, optional): Confusion matrix of the initial (naive) Random Forest model.
        confusion_matrix_percent (ndarray, optional): Percent-based confusion matrix for the initial (naive) model.
        tuned_accuracy (float, optional): Accuracy of the tuned Random Forest model.
        tuned_confusion_matrix (ndarray, optional): Confusion matrix of the tuned Random Forest model.
        tuned_confusion_matrix_percent (ndarray, optional): Percent-based confusion matrix for the tuned model.

    """
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initializes the RandomForest class with the provided training and testing data.

        Args:
            X_train (ndarray or DataFrame): The training features for model fitting.
            y_train (ndarray or Series): The training labels for model fitting.
            X_test (ndarray or DataFrame): The test features for model validation.
            y_test (ndarray or Series): The test labels for model validation.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.initial_rf = None
        self.tuned_rf = None

        # Automatically run the initial model 
        self.model()

    # Initial model and validation
    def model(self, n_estimators=100,**kwargs):
        """
        Trains a random forest classifier with the provided hyperparameters and validates it.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Default is 100.
            **kwargs: Additional keyword arguments passed to the RandomForestClassifier.

        Returns:
            RandomForestClassifier: The trained (naive) random forest classifier.

        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        # Initialize model and fit the model
        rf = RandomForestClassifier(n_estimators= n_estimators, **kwargs)
        rf.fit(self.X_train, self.y_train)
        
        # Validate the initial model and return validation metrics
        y_pred = rf.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        self.confusion_matrix_percent = self.confusion_matrix.astype(float) / self.confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.classification_report = classification_report(self.y_test, y_pred)

        self.initial_rf = rf
        return self.initial_rf
    
    # Tune the best parameters for classifier using random search or grid search methods
    def tune(self, method="random", n_estimators=[100, 200, 300, 500, 1000], max_depth=[None, 10, 20, 30, 50], min_samples_split=[2, 5, 10, 20], min_samples_leaf=[1, 2, 3, 5], max_features= ['sqrt'], n_iter=5, cv=5, n_job=-1):
        """
        Tunes the Random Forest model's hyperparameters using grid or random search.

        Args:
            method (str, optional): The method used for hyperparameter search. Can be 'random' or 'grid'. Default is 'random'.
            n_estimators (list, optional): List of values for the number of trees to search over. Default is [100, 200, 300, 500, 1000].
            max_depth (list, optional): List of values for the maximum depth of trees. Default is [None].
            min_samples_split (list, optional): List of values for the minimum number of samples required to split an internal node. Default is [2, 5, 10, 20].
            min_samples_leaf (list, optional): List of values for the minimum number of samples required to be at a leaf node. Default is [1, 2, 3, 5].
            max_features (list, optional): List of values for the number of features to consider when looking for the best split. Default is ['sqrt'].
            n_iter (int, optional): The number of iterations for RandomizedSearchCV. Default is 3.
            cv (int, optional): Cross-validation generator or an iterable. Default is 3.
            n_jobs (int, optional): The number of jobs to run in parallel. Default is -1 (use all processors).

        Returns:
            RandomForestClassifier: The tuned Random Forest classifier.

        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        paras = [{
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }]
        
        if method.lower() == 'random' or method.lower() =='randomized' or method.lower() =='randomizedsearch' or method.lower() =='randomizedsearchcv':
            random_searched = RandomizedSearchCV(estimator= self.initial_rf, param_distributions=paras, n_iter= n_iter, scoring='accuracy', verbose=True)
            random_searched.fit(self.X_train, self.y_train)
            tuned_model = random_searched
            self.tuned_rf = tuned_model

        elif method.lower() == 'grid' or method.lower() == 'gridsearch' or method.lower() == 'gridsearchcv':
            grid_search = GridSearchCV(estimator= self.initial_rf, param_grid= paras, cv=cv, scoring='accuracy', verbose=True, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)

            tuned_model = grid_search
            self.tuned_rf = tuned_model

        else:
            raise ValueError('Tune method is not supported, the current methods are "randomizedsearch" and "gridsearchcv"')
        
        # Validate the initial model and return validation metrics
        tunded_y_pred = tuned_model.predict(self.X_test)
        self.tuned_accuracy = accuracy_score(self.y_test, tunded_y_pred)
        self.tuned_confusion_matrix = confusion_matrix(self.y_test, tunded_y_pred)
        self.tuned_confusion_matrix_percent = self.tuned_confusion_matrix.astype(float) / self.tuned_confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.tuned_classification_report = classification_report(self.y_test, tunded_y_pred)

        return self.tuned_rf
    
    # Classify image 
    def classify(self, src, model=None):
        """
        Classifies an image using the trained or tuned Random Forest model.

        Args:
            src (rasterio.DatasetReader): A rasterio object representing the image to classify.
            model (object): trained Random forest model to classify image.

        Returns:
            rasterio.DatasetReader: The classified image as a raster object.

        """
        import rasterio
        from .common import reshape_raster, array2raster

        # Define the model to use
        if model is not None:
            RF_model = model
        else: 
            RF_model = self.tuned_rf if self.tuned_rf is not None else self.initial_rf
        
        # Define input parameters
        if not isinstance(src, rasterio.DatasetReader):
            raise ValueError('Source image is not supported')
        else: 
            src_meta = src.meta
            nbands = src.count
            src_height = src.height
            src_width = src.width            
            src_rast = src.read()
            
            # Reshape and flatten data
            src_img = reshape_raster(src_rast, mode='image')
            ds = src_img.reshape((-1, nbands))
            
            # Predict labels using the define model
            pred_labels = RF_model.predict(ds)

            # Reshape data and convert to raster format
            pred_result = pred_labels.reshape(src_height, src_width)

            src_meta.update({'count': 1})
            classified = array2raster(pred_result, metadata=src_meta)            

            return classified
        

# =========================================================================================== #
#              Train, Tune, and Classify image using Support Vector Machine
# =========================================================================================== #   
class SVM:
    """
    A class that encapsulates a Support Vector Machine (SVM) model for classification tasks, including model training,
    hyperparameter tuning using grid or random search, and classification of image data.

    Attributes:
        X_train (ndarray or DataFrame): The training features for model fitting.
        y_train (ndarray or Series): The training labels for model fitting.
        X_test (ndarray or DataFrame): The test features for model validation.
        y_test (ndarray or Series): The test labels for model validation.
        initial_svm (SVC, optional): The initial SVM model (untuned).
        tuned_svm (SVC, optional): The SVM model after tuning using grid or random search.
        accuracy (float, optional): Accuracy of the initial (naive) SVM model.
        confusion_matrix (ndarray, optional): Confusion matrix of the initial (naive) SVM model.
        confusion_matrix_percent (ndarray, optional): Percent-based confusion matrix for the initial (naive) model.
        tuned_accuracy (float, optional): Accuracy of the tuned SVM model.
        tuned_confusion_matrix (ndarray, optional): Confusion matrix of the tuned SVM model.
        tuned_confusion_matrix_percent (ndarray, optional): Percent-based confusion matrix for the tuned model.
    
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initializes the SVM class with the provided training and testing data.

        Args:
            X_train (ndarray or DataFrame): The training features for model fitting.
            y_train (ndarray or Series): The training labels for model fitting.
            X_test (ndarray or DataFrame): The test features for model validation.
            y_test (ndarray or Series): The test labels for model validation.
        
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.initial_svm = None
        self.tuned_svm = None

        # Automatically run the initial model 
        self.model()

    # Initial model and validation
    def model(self, kernel='rbf',**kwargs):
        """
        Trains and validates the initial SVM model.

        Args:
            kernel (str): Specifies the kernel type to be used in the algorithm. Default is 'rbf'.
            **kwargs: Additional keyword arguments for the SVC model.

        Returns:
            SVC: The trained SVM model.
        
        """
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        # Initialize model and fit the model
        svm = SVC(kernel=kernel, **kwargs)
        svm.fit(self.X_train, self.y_train)
        
        # Validate the initial model and return validation metrics
        y_pred = svm.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        self.confusion_matrix_percent = self.confusion_matrix.astype(float) / self.confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.classification_report = classification_report(self.y_test, y_pred)

        self.initial_svm = svm
        return self.initial_svm
    
    # Tune the best parameters for classifier using random search or grid search methods
    def tune(self, method="random", kernel=['rbf'], C=[1, 2, 4, 8, 10, 16, 32, 64, 100, 128, 1000], gamma=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0], n_iter=5, cv=5, n_job=-1):
        """
        Tunes the best parameters for the SVM classifier using random search or grid search methods.

        Args:
            method (str): The tuning method to use ('random' or 'grid'). Default is 'random'.
            kernel (list): List of kernel types to be used in the algorithm. Default is ['rbf'].
            C (list): List of regularization parameters. Default is [1, 2, 4, 8, 10, 16, 32, 64, 100, 128, 1000].
            gamma (list): List of kernel coefficient values. Default is [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0].
            n_iter (int): Number of parameter settings that are sampled in random search. Default is 3.
            cv (int): Number of cross-validation folds. Default is 5.
            n_job (int): Number of jobs to run in parallel. Default is -1 (use all processors).

        Returns:
            SVC: The tuned SVM model.
        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        paras = [{
            'kernel': kernel,
            'C': C,
            'gamma': gamma
        }]
        
        if method.lower() == 'random' or method.lower() =='randomized' or method.lower() =='randomizedsearch' or method.lower() =='randomizedsearchcv':
            random_searched = RandomizedSearchCV(estimator= self.initial_svm, param_distributions=paras, n_iter= n_iter, scoring='accuracy', verbose=True)
            random_searched.fit(self.X_train, self.y_train)
            tuned_model = random_searched
            self.tuned_svm = tuned_model

        elif method.lower() == 'grid' or method.lower() == 'gridsearch' or method.lower() == 'gridsearchcv':
            grid_search = GridSearchCV(estimator= self.initial_svm, param_grid= paras, cv=cv, n_jobs=-1, scoring='accuracy', verbose=True)
            grid_search.fit(self.X_train, self.y_train)

            tuned_model = grid_search
            self.tuned_svm = tuned_model

        else:
            raise ValueError('Tune method is not supported, the current methods are "randomizedsearch" and "gridsearchcv"')
        
        # Validate the initial model and return validation metrics
        tuned_y_pred = tuned_model.predict(self.X_test)
        self.tuned_accuracy = accuracy_score(self.y_test, tuned_y_pred)
        self.tuned_confusion_matrix = confusion_matrix(self.y_test, tuned_y_pred)
        self.tuned_confusion_matrix_percent = self.tuned_confusion_matrix.astype(float) / self.tuned_confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.tuned_classification_report = classification_report(self.y_test, tuned_y_pred)

        return self.tuned_svm
    
    # Classify image 
    def classify(self, src, model=None):
        """
        Classifies an image using the trained or tuned SVM model.

        Args:
            src (rasterio.DatasetReader): A rasterio object representing the image to classify.
            model (SVC, optional): Trained SVM model to classify the image. If None, uses the tuned SVM model if available, otherwise the naive SVM model.

        Returns:
            rasterio.DatasetReader: The classified image as a raster object.

        """
        import rasterio
        from .common import reshape_raster, array2raster

        # Define the model to use
        if model is not None:
            SVM_model = model
        else: 
            SVM_model = self.tuned_svm if self.tuned_svm is not None else self.initial_svm
        
        # Define input parameters
        if not isinstance(src, rasterio.DatasetReader):
            raise ValueError('Source image is not supported')
        else: 
            src_meta = src.meta
            nbands = src.count
            src_height = src.height
            src_width = src.width            
            src_rast = src.read()
            
            # Reshape and flatten data
            src_img = reshape_raster(src_rast, mode='image')
            ds = src_img.reshape((-1, nbands))
            
            # Predict labels using the define model
            pred_labels = SVM_model.predict(ds)

            # Reshape data and convert to raster format
            pred_result = pred_labels.reshape(src_height, src_width)

            src_meta.update({'count': 1})
            classified = array2raster(pred_result, metadata=src_meta)            

            return classified
        
# =========================================================================================== #
#              Train, Tune, and Classify image using KNeighborsClassifier 
# =========================================================================================== #  
class KNN:
    """
    A class that encapsulates a K-Nearest Neighbors (KNN) model for classification tasks, including model training,
    hyperparameter tuning using grid or random search, and classification of image data.

    Attributes:
        X_train (ndarray or DataFrame): The training features for model fitting.
        y_train (ndarray or Series): The training labels for model fitting.
        X_test (ndarray or DataFrame): The test features for model validation.
        y_test (ndarray or Series): The test labels for model validation.
        initial_knn (KNeighborsClassifier, optional): The initial KNN model (untuned).
        tuned_knn (KNeighborsClassifier, optional): The KNN model after tuning using grid or random search.
        accuracy (float, optional): Accuracy of the initial (naive) KNN model.
        confusion_matrix (ndarray, optional): Confusion matrix of the initial (naive) KNN model.
        confusion_matrix_percent (ndarray, optional): Percent-based confusion matrix for the initial (naive) model.
        tuned_accuracy (float, optional): Accuracy of the tuned KNN model.
        tuned_confusion_matrix (ndarray, optional): Confusion matrix of the tuned KNN model.
        tuned_confusion_matrix_percent (ndarray, optional): Percent-based confusion matrix for the tuned model.

    """
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initializes the KNN class with the provided training and testing data.

        Args:
            X_train (ndarray or DataFrame): The training features for model fitting.
            y_train (ndarray or Series): The training labels for model fitting.
            X_test (ndarray or DataFrame): The test features for model validation.
            y_test (ndarray or Series): The test labels for model validation.

        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.initial_knn = None
        self.tuned_knn = None

        # Automatically run the initial model 
        self.model()

    # Initial model and validation
    def model(self, **kwargs):
        """
        Trains and validates the initial KNN model.

        Args:
            **kwargs: Additional keyword arguments for the KNeighborsClassifier model.

        Returns:
            KNeighborsClassifier: The trained KNN model.

        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        # Initialize model and fit the model
        knn = KNeighborsClassifier(**kwargs)
        knn.fit(self.X_train, self.y_train)
        
        # Validate the initial model and return validation metrics
        y_pred = knn.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        self.confusion_matrix_percent = self.confusion_matrix.astype(float) / self.confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.classification_report = classification_report(self.y_test, y_pred)

        self.initial_knn = knn
        return self.initial_knn
    
    # Tune the best parameters for classifier using random search or grid search methods
    def tune(self, method="random", n_neighbors=[3, 5, 7, 9, 11], weights= ['uniform', 'distance'],n_iter=5, cv=5, n_job=-1):
        """
        Tunes the best parameters for the KNN classifier using random search or grid search methods.

        Args:
            method (str): The tuning method to use ('random' or 'grid'). Default is 'random'.
            n_neighbors (list): List of values for the number of neighbors to use. Default is [3, 5, 7, 9, 11].
            weights (list): List of weight functions used in prediction. Default is ['uniform', 'distance'].
            n_iter (int): Number of parameter settings that are sampled in random search. Default is 5.
            cv (int): Number of cross-validation folds. Default is 5.
            n_jobs (int): Number of jobs to run in parallel. Default is -1 (use all processors).

        Returns:
            KNeighborsClassifier: The tuned KNN model.

        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        paras = [{
            'n_neighbors': n_neighbors,
            'weights': weights
        }]
        
        if method.lower() == 'random' or method.lower() =='randomized' or method.lower() =='randomizedsearch' or method.lower() =='randomizedsearchcv':
            random_searched = RandomizedSearchCV(estimator= self.initial_knn, param_distributions=paras, n_iter= n_iter, scoring='accuracy', verbose=True)
            random_searched.fit(self.X_train, self.y_train)
            tuned_model = random_searched
            self.tuned_knn = tuned_model

        elif method.lower() == 'grid' or method.lower() == 'gridsearch' or method.lower() == 'gridsearchcv':
            grid_search = GridSearchCV(estimator= self.initial_knn, param_grid= paras, cv=cv, verbose=True, n_jobs=-1, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)

            tuned_model = grid_search
            self.tuned_knn = tuned_model

        else:
            raise ValueError('Tune method is not supported, the current methods are "randomizedsearch" and "gridsearchcv"')
        
        # Validate the initial model and return validation metrics
        tuned_y_pred = tuned_model.predict(self.X_test)
        self.tuned_accuracy = accuracy_score(self.y_test, tuned_y_pred)
        self.tuned_confusion_matrix = confusion_matrix(self.y_test, tuned_y_pred)
        self.tuned_confusion_matrix_percent = self.tuned_confusion_matrix.astype(float) / self.tuned_confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.tuned_classification_report = classification_report(self.y_test, tuned_y_pred)

        return self.tuned_knn
    
    # Classify image 
    def classify(self, src, model=None):
        """
        Classifies an image using the trained or tuned KNN model.

        Args:
            src (rasterio.DatasetReader): A rasterio object representing the image to classify.
            model (KNeighborsClassifier, optional): Trained KNN model to classify the image. If None, uses the tuned KNN model if available, otherwise the naive KNN model.

        Returns:
            rasterio.DatasetReader: The classified image as a raster object.

        """
        import rasterio
        from geonate.common import reshape_raster, array2raster

        # Define the model to use
        if model is not None:
            KNN_model = model
        else: 
            KNN_model = self.tuned_knn if self.tuned_knn is not None else self.initial_knn
        
        # Define input parameters
        if not isinstance(src, rasterio.DatasetReader):
            raise ValueError('Source image is not supported')
        else: 
            src_meta = src.meta
            nbands = src.count
            src_height = src.height
            src_width = src.width            
            src_rast = src.read()
            
            # Reshape and flatten data
            src_img = reshape_raster(src_rast, mode='image')
            ds = src_img.reshape((-1, nbands))
            
            # Predict labels using the define model
            pred_labels = KNN_model.predict(ds)

            # Reshape data and convert to raster format
            pred_result = pred_labels.reshape(src_height, src_width)

            src_meta.update({'count': 1})
            classified = array2raster(pred_result, metadata=src_meta)            

            return classified
        

# =========================================================================================== #
#              Train, Tune, and Classify image using Gaussian Naive Bayes
# =========================================================================================== #   
class Gaussian_Naive_Bayes:
    """
    A class to implement Gaussian Naive Bayes classifier for training, tuning, and classification tasks.

    """
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialize the Gaussian_Naive_Bayes class with training and testing data.

        Args:
            X_train (array-like): Training feature data.
            y_train (array-like): Training target data.
            X_test (array-like): Testing feature data.
            y_test (array-like): Testing target data.

        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.initial_gnb = None
        self.tuned_gnb = None

        # Automatically run the initial model 
        self.model()

    # Initial model and validation
    def model(self, **kwargs):
        """
        Train and validate the initial Gaussian Naive Bayes model.

        Args:
            **kwargs: Additional keyword arguments for GaussianNB.

        Returns:
            GaussianNB: The trained Gaussian Naive Bayes model.

        """
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        # Initialize model and fit the model
        gnb = GaussianNB(**kwargs)
        gnb.fit(self.X_train, self.y_train)
        
        # Validate the initial model and return validation metrics
        y_pred = gnb.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        self.confusion_matrix_percent = self.confusion_matrix.astype(float) / self.confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.classification_report = classification_report(self.y_test, y_pred)

        self.initial_gnb = gnb
        return self.initial_gnb
    
    # Tune the best parameters for classifier using random search or grid search methods
    def tune(self, method="random", var_smoothing=[1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0], n_iter=5, cv=5, n_job=-1):
        """
        Tune the Gaussian Naive Bayes model using random search or grid search methods.

        Args:
            method (str): The tuning method to use ('random' or 'grid').
            var_smoothing (list): List of var_smoothing values to try.
            n_iter (int): Number of iterations for random search.
            cv (int): Number of cross-validation folds.
            n_job (int): Number of jobs to run in parallel.

        Returns:
            The tuned Gaussian Naive Bayes model.

        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        paras = [{
            'var_smoothing': var_smoothing
        }]
        
        if method.lower() == 'random' or method.lower() =='randomized' or method.lower() =='randomizedsearch' or method.lower() =='randomizedsearchcv':
            random_searched = RandomizedSearchCV(estimator= self.initial_gnb, param_distributions=paras, n_iter= n_iter, scoring='accuracy', verbose=True)
            random_searched.fit(self.X_train, self.y_train)
            tuned_model = random_searched
            self.tuned_gnb = tuned_model

        elif method.lower() == 'grid' or method.lower() == 'gridsearch' or method.lower() == 'gridsearchcv':
            grid_search = GridSearchCV(estimator= self.initial_gnb, param_grid= paras, cv=cv, scoring='accuracy', verbose=True, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)

            tuned_model = grid_search
            self.tuned_gnb = tuned_model

        else:
            raise ValueError('Tune method is not supported, the current methods are "randomizedsearch" and "gridsearchcv"')
        
        # Validate the initial model and return validation metrics
        tuned_y_pred = tuned_model.predict(self.X_test)
        self.tuned_accuracy = accuracy_score(self.y_test, tuned_y_pred)
        self.tuned_confusion_matrix = confusion_matrix(self.y_test, tuned_y_pred)
        self.tuned_confusion_matrix_percent = self.tuned_confusion_matrix.astype(float) / self.tuned_confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.tuned_classification_report = classification_report(self.y_test, tuned_y_pred)

        return self.tuned_gnb
    
    # Classify image 
    def classify(self, src, model=None):
        """
        Classify an image using the Gaussian Naive Bayes model.

        Args:
            src (rasterio.DatasetReader): The source image to classify.
            model (GaussianNB, optional): The model to use for classification. If None, the tuned model or initial model will be used.

        Returns:
            The classified image.
            
        """
        import rasterio
        from geonate.common import reshape_raster, array2raster

        # Define the random forest model to use
        if model is not None:
            GNB_model = model
        else: 
            GNB_model = self.tuned_gnb if self.tuned_gnb is not None else self.initial_gnb
        
        # Define input parameters
        if not isinstance(src, rasterio.DatasetReader):
            raise ValueError('Source image is not supported')
        else: 
            src_meta = src.meta
            nbands = src.count
            src_height = src.height
            src_width = src.width            
            src_rast = src.read()
            
            # Reshape and flatten data
            src_img = reshape_raster(src_rast, mode='image')
            ds = src_img.reshape((-1, nbands))
            
            # Predict labels using the defined model
            pred_labels = GNB_model.predict(ds)

            # Reshape data and convert to raster format
            pred_result = pred_labels.reshape(src_height, src_width)

            src_meta.update({'count': 1})
            classified = array2raster(pred_result, metadata=src_meta)            

            return classified

# =========================================================================================== #
#              Train, Tune, and Classify image using XGBoost
# =========================================================================================== #  
class XGBoost:
    """
    A wrapper class for XGBoost classification, including model training, hyperparameter tuning, and classification of raster images.

    Attributes:
        X_train (array-like): Training feature set.
        y_train (array-like): Training labels.
        X_test (array-like): Testing feature set.
        y_test (array-like): Testing labels.
        initial_xgb (XGBClassifier or None): The initial trained XGBoost model.
        tuned_xgb (XGBClassifier or None): The tuned XGBoost model (if tuning is performed).
        accuracy (float): Accuracy of the initial model on the test set.
        confusion_matrix (ndarray): Confusion matrix of the initial model.
        confusion_matrix_percent (ndarray): Normalized confusion matrix as percentages.
        classification_report (str): Classification report for the initial model.
        tuned_accuracy (float): Accuracy of the tuned model.
        tuned_confusion_matrix (ndarray): Confusion matrix of the tuned model.
        tuned_confusion_matrix_percent (ndarray): Normalized confusion matrix for the tuned model.
        tuned_classification_report (str): Classification report for the tuned model.
        
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initializes the XGBoost classifier with training and testing data, and automatically trains an initial model.

        Args:
            X_train (array-like): Training feature set.
            y_train (array-like): Training labels.
            X_test (array-like): Testing feature set.
            y_test (array-like): Testing labels.

        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.initial_xgb = None
        self.tuned_xgb = None

        # Automatically run the initial model 
        self.model()

    # Initial model and validation
    def model(self, **kwargs):
        """
        Trains an initial XGBoost classifier using the provided training data and evaluates its performance on the test set.

        Args:
            **kwargs: Additional parameters to pass to XGBClassifier.

        Returns:
            XGBClassifier: The trained initial model.

        """
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        # Initialize model and fit the model
        xgb = XGBClassifier(**kwargs)
        xgb.fit(self.X_train, self.y_train)
        
        # Validate the initial model and return validation metrics
        y_pred = xgb.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        self.confusion_matrix_percent = self.confusion_matrix.astype(float) / self.confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.classification_report = classification_report(self.y_test, y_pred)

        self.initial_xgb = xgb
        return self.initial_xgb
    
    # Tune the best parameters for classifier using random search or grid search methods
    def tune(self, method="random", n_estimators=[100, 200, 300, 500, 1000], max_depth=[3, 5, 7, 9], learning_rate=[0.0001, 0.001, 0.01, 0.1], subsample=[0.5, 0.7, 1], n_iter=5, cv=5, n_job=-1):
        """
        Tunes the hyperparameters of the XGBoost classifier using either RandomizedSearchCV or GridSearchCV.

        Args:
            method (str, optional): Search method, either "random" (default) or "grid".
            n_estimators (list, optional): List of values for the number of trees.
            max_depth (list, optional): List of values for the maximum tree depth.
            learning_rate (list, optional): List of learning rates.
            subsample (list, optional): List of subsampling ratios.
            n_iter (int, optional): Number of iterations for random search (ignored for grid search).
            cv (int, optional): Number of cross-validation folds.
            n_job (int, optional): Number of parallel jobs (currently not used in the function).

        Returns:
            Best estimator from tuning process (XGBClassifier).

        """
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        paras = [{
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample
        }]
        
        if method.lower() == 'random' or method.lower() =='randomized' or method.lower() =='randomizedsearch' or method.lower() =='randomizedsearchcv':
            random_searched = RandomizedSearchCV(estimator= self.initial_xgb, param_distributions=paras, n_iter= n_iter, scoring='accuracy', verbose=True)
            random_searched.fit(self.X_train, self.y_train)
            tuned_model = random_searched
            self.tuned_xgb = tuned_model

        elif method.lower() == 'grid' or method.lower() == 'gridsearch' or method.lower() == 'gridsearchcv':
            grid_search = GridSearchCV(estimator= self.initial_xgb, param_grid= paras, cv=cv, scoring='accuracy', verbose=True, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)

            tuned_model = grid_search
            self.tuned_xgb = tuned_model

        else:
            raise ValueError('Tune method is not supported, the current methods are "randomizedsearch" and "gridsearchcv"')
        
        # Validate the initial model and return validation metrics
        tuned_y_pred = tuned_model.predict(self.X_test)
        self.tuned_accuracy = accuracy_score(self.y_test, tuned_y_pred)
        self.tuned_confusion_matrix = confusion_matrix(self.y_test, tuned_y_pred)
        self.tuned_confusion_matrix_percent = self.tuned_confusion_matrix.astype(float) / self.tuned_confusion_matrix.sum(axis=1, keepdims=True) * 100
        self.tuned_classification_report = classification_report(self.y_test, tuned_y_pred)

        return self.tuned_xgb
    
    # Classify image 
    def classify(self, src, model=None):
        """
        Classifies an input raster image using the trained XGBoost model.

        Args:
            src (rasterio.DatasetReader): The source raster image to classify.
            model (XGBClassifier, optional): The model to use for classification. If not provided, the tuned model is used (or the initial model if tuning was not performed).

        Returns:
            rasterio.io.MemoryFile: The classified raster image.
            
        """
        import rasterio
        from geonate.common import reshape_raster, array2raster

        # Define the random forest model to use
        if model is not None:
            XGB_model = model
        else: 
            XGB_model = self.tuned_xgb if self.tuned_xgb is not None else self.initial_xgb
        
        # Define input parameters
        if not isinstance(src, rasterio.DatasetReader):
            raise ValueError('Source image is not supported')
        else: 
            src_meta = src.meta
            nbands = src.count
            src_height = src.height
            src_width = src.width            
            src_rast = src.read()
            
            # Reshape and flatten data
            src_img = reshape_raster(src_rast, mode='image')
            ds = src_img.reshape((-1, nbands))
            
            # Predict labels using the defined model
            pred_labels = XGB_model.predict(ds)

            # Reshape data and convert to raster format
            pred_result = pred_labels.reshape(src_height, src_width)

            src_meta.update({'count': 1})
            classified = array2raster(pred_result, metadata=src_meta)            

            return classified
