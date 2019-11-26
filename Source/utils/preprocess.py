import numpy as np
from sklearn.cluster import KMeans


class Preprocess:

    def __init__(self):
        self.attr_names = None
        self.sol_cols = None
        self.attr_values = {}
        self.attr_types = {}
        self.models = None

    def extract_attr_info(self, data, num_class):
        # Extract the column names of the attributes
        columns = data.columns.values.tolist()
        self.attr_names = columns[:-num_class]
        self.sol_cols = columns[-num_class:]
        # self.attr_names.remove('class')

        # For each column extract the possible values
        for attr_ix in range(len(self.attr_names)):
            attr_values = np.array(data[self.attr_names[attr_ix]].unique())

            if (len(attr_values) > 20 and columns[attr_ix] != 'medical_specialty' and columns[attr_ix] !=
                'discharge_disposition_id') or (columns[attr_ix] == 'time_in_hospital' or columns[attr_ix] ==
                'number_diagnoses' or columns[attr_ix] == 'num_procedures' or columns[attr_ix] == 'number_emergency' or columns[attr_ix] == 'number_inpatient'):
                # being selected as numerical attrobute
                # print("Assigning attribute " + self.attr_names[attr_ix] + " to NUMERICAL")
                self.attr_types[attr_ix] = "num_continuous"
                self.attr_values[attr_ix] = []
            else:
                # print("Assigning attribute " + self.attr_names[attr_ix] + " to CATEGORICAL")
                self.attr_types[attr_ix] = "categorical"
                if columns[attr_ix] == 'admission_source_id':
                    attr_values = np.concatenate((attr_values, np.array([13])), axis=0)  # Sometimes it does not
                    # appear in the training set
                self.attr_values[attr_ix] = attr_values

        return self.attr_names, self.attr_values, self.attr_types, self.sol_cols, columns

    def fit_predict(self, data, columns_names, n_clusters):
        self.models = [None] * data.shape[1]
        aux_data = np.copy(data)

        for i in range(len(self.attr_names)):
            if self.attr_types[i] == 'num_continuous':

                print('\033[1mDISCRETIZATION of attribute: \033[0m' + str(columns_names[i]))
                km = KMeans(n_clusters=n_clusters)
                km.fit(aux_data[:, i].reshape(-1, 1))
                self.models[i] = km
                aux_data[:, i] = km.predict(aux_data[:, i].reshape(-1, 1))
                self.attr_values[i] = np.array(range(n_clusters))

                # print('The cluster centers obtained for the variable \033[1m' + self.attr_names[i] + '\033[0m are:')
                # for c in range(n_clusters):
                #     print('Cluster ' + str(c) + ': ' + str(km.cluster_centers_[c]))

        return aux_data, self.attr_values

    def predict(self, instance):
        pred_instance = np.copy(instance)

        for i in range(len(self.attr_names)):
            if self.attr_types[i] == 'num_continuous':
                pred_instance[i] = self.models[i].predict(instance[i])

        return pred_instance
