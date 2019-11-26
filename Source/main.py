import numpy as np
import pandas as pd
import pickle
from Source.case_base import CaseBase
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------

# Paths
DATA_PATH = '../Data/diabetic_data.csv'

# CBR
MODE = 'tests'

# Do not modify
NUM_ATTRIB_SOLUTION = 23


# ----------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # For debugging purposes
    np.random.seed(3)

    # Read file and delete nans by the mode of the attribute
    print("---LOADING DATASET & PREPROCESSING---")
    data = pd.read_csv('../Data/diabetic_data.csv', na_values='?', low_memory=False)
    for col in data.columns:
        if data[col].isna().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Delete cases where the patient did not come again (select successes)
    b = data['readmitted'] == 'NO'
    data = data[b]
    data = data.drop(['readmitted','encounter_id','patient_nbr','weight'], axis=1)
    cols_to_move = ['change', 'diabetesMed']
    data = data[cols_to_move + [col for col in data.columns if col not in cols_to_move]]

    # Discretize diagnosis using expert knowledge (paper)
    special_cases = ['diag_1', 'diag_2', 'diag_3']
    mapping = {}
    for i in tqdm(range(len(special_cases))):
        diags_unique = np.unique(data[special_cases[i]].values[~pd.isna(data[special_cases[i]].values)])
        for str_val in diags_unique:
            if str_val not in mapping:
                try:
                    val = float(str_val)
                    if (val >= 390 and val <= 459) or (val == 785):
                        mapping[str_val] = 0
                    elif (val >= 460 and val <= 519) or (val == 786):
                        mapping[str_val] = 1
                    elif (val >= 520 and val <= 579) or (val == 787):
                        mapping[str_val] = 2
                    elif (val >= 250 and val <= 251):
                        mapping[str_val] = 3
                    elif (val >= 800 and val <= 999):
                        mapping[str_val] = 4
                    elif (val >= 710 and val <= 739):
                        mapping[str_val] = 5
                    elif (val >= 580 and val <= 629) or (val == 788):
                        mapping[str_val] = 6
                    elif (val >= 140 and val <= 239):
                        mapping[str_val] = 7
                    elif (val >= 240 and val <= 279 and val != 250):
                        mapping[str_val] = 9
                    elif (val >= 680 and val <= 709) or (val == 782):
                        mapping[str_val] = 10
                    elif (val >= 1 and val <= 139):
                        mapping[str_val] = 11
                    elif (val >= 290 and val <= 319):
                        mapping[str_val] = 12
                    elif (val >= 280 and val <= 289):
                        mapping[str_val] = 14
                    elif (val >= 320 and val <= 359):
                        mapping[str_val] = 15
                    elif (val >= 630 and val <= 679):
                        mapping[str_val] = 16
                    elif (val >= 360 and val <= 389):
                        mapping[str_val] = 17
                    elif (val >= 740 and val <= 759):
                        mapping[str_val] = 18
                    else:
                        mapping[str_val] = 8
                except ValueError:
                    mapping[str_val] = 13
        data[special_cases[i]] = data[special_cases[i]].replace(mapping)

    # Discretize diagnosis using expert knowledge (paper)
    special_cases = data.columns.values[-NUM_ATTRIB_SOLUTION:]
    mapping = {'No':'No', 'Steady':'Yes', 'Up':'Yes', 'Down':'Yes'}
    for i in range(len(special_cases)):
        data[special_cases[i]] = data[special_cases[i]].replace(mapping)

    # Saving the objects:
    # data.to_pickle('../Data/dataobject.pkl')

    # Getting back the objects:
    # data = pd.read_pickle('../Data/dataobject.pkl')

    # Split train-test
    msk = np.random.rand(len(data)) < 0.7
    data_train = data[msk]
    data_test = data[~msk]

    # Run tests for the CBR
    if MODE == 'tests':
        # print("---TRAINING---")
        # cb = CaseBase(data_train, NUM_ATTRIB_SOLUTION)
        # pickle_out = open("../Data/CBpickle", "wb")
        # pickle.dump(cb, pickle_out)
        # pickle_out.close()

        print("---LOADING DECISION TREE---")
        pickle_in = open("../Data/CBpickle", "rb")
        cb = pickle.load(pickle_in)

        run_tests(cb, data_test)


# ----------------------------------------------------------------------------------------------------------------------
# RUN AN INSTANCE OVER THE DT
# ----------------------------------------------------------------------------------------------------------------------
def run(cb, new_case, num_attrib_solution):
    # [R1]: RETRIEVE ****
    retrieved_cases = cb.retrieve_v2(new_case)
    # [R2]: REUSE ****
    solution = cb.update(retrieved_cases, new_case, num_attrib_solution)
    return solution


# ----------------------------------------------------------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------------------------------------------------------
def run_tests(cb, data_test):
    # Distances between returned songs and target
    print("---TESTING---")
    for idx in tqdm(range(len(data_test))):
        data = data_test.iloc[idx].values[:-NUM_ATTRIB_SOLUTION]
        solution = run(cb, data, NUM_ATTRIB_SOLUTION)
        # print(data)
        # print(solution)
        # print("------------------------------------------------------------------------------")

    print("Do not use MODE because will be None")


# ----------------------------------------------------------------------------------------------------------------------
# RUN MAIN
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
