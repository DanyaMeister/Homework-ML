import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Generate random bootstrap samples (with replacement)
            bag_indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(bag_indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = data
        self.target = target
        self._generate_splits(data)
        
        # Check if all bags have the same length
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()  # Instantiate the model (without parameters)
            data_bag, target_bag = data[self.indices_list[bag]], target[self.indices_list[bag]]  # Bootstrap sample
            self.models_list.append(model.fit(data_bag, target_bag))  # Train model and store
            
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Predict using the average of all models
        return np.mean([model.predict(data) for model in self.models_list], axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        
        for i in range(len(self.data)):
            for bag in range(self.num_bags):
                if i not in self.indices_list[bag]:  # Check if the current data point was not in this bag
                    model = self.models_list[bag]
                    prediction = model.predict([self.data[i]])[0]  # Get prediction for this data point
                    list_of_predictions_lists[i].append(prediction)
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = []  # List to hold final OOB predictions
        
        for predictions in self.list_of_predictions_lists:
            if len(predictions) < self.num_bags:
                average_prediction = np.mean(predictions, axis=0)
                self.oob_predictions.append(average_prediction)
            else:
                self.oob_predictions.append(None)
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        squared_errors = []
        
        for i in range(len(self.data)):
            prediction = self.oob_predictions[i]
            if prediction is not None:
                squared_error = ((prediction - self.target[i]) ** 2)
                if not np.isnan(squared_error).any():
                    squared_errors.append(squared_error)

        mean_squared_error = np.mean(squared_errors, axis=0)
        return mean_squared_error  # Return the mean squared error
