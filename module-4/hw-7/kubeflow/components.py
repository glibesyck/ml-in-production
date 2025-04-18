from kfp.dsl import component, Output, Dataset, Model, Input

@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def prepare_data(df_data: Output[Dataset]):
    import pandas as pd
    import os
    from sklearn import datasets
    
    # Load dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df = df.dropna()

    if not os.path.exists(df_data.path):
        os.makedirs(df_data.path)

    df.to_csv(f'{df_data.path}/final_df.csv', index=False)


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def train_test_split(df_data: Input[Dataset], npy_data: Output[Dataset]):    
    import pandas as pd
    import numpy as np
    import os
    from sklearn.model_selection import train_test_split
    
    final_data = pd.read_csv(f'{df_data.path}/final_df.csv')
    
    target_column = 'species'
    X = final_data.loc[:, final_data.columns != target_column]
    y = final_data.loc[:, final_data.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify = y, random_state=47)

    if not os.path.exists(npy_data.path):
        os.makedirs(npy_data.path)
    
    np.save(f'{npy_data.path}/X_train.npy', X_train)
    np.save(f'{npy_data.path}/X_test.npy', X_test)
    np.save(f'{npy_data.path}/y_train.npy', y_train)
    np.save(f'{npy_data.path}/y_test.npy', y_test)

@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def training_basic_classifier(npy_data: Input[Dataset], model: Output[Model]):
    import numpy as np
    import os
    from sklearn.linear_model import LogisticRegression
    import pickle
    
    X_train = np.load(f'{npy_data.path}/X_train.npy',allow_pickle=True)
    y_train = np.load(f'{npy_data.path}/y_train.npy',allow_pickle=True)
    
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train,y_train)

    if not os.path.exists(model.path):
        os.makedirs(model.path)

    with open(f'{model.path}/model.pkl', "wb") as f:
        pickle.dump(classifier, f)


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
    output_component_file='predict.yaml'
)
def predict_on_test_data(model: Input[Model], npy_data: Input[Dataset], y_pred_file: Output[Dataset]):
    import pandas as pd
    import numpy as np
    import pickle
    import os
    import shutil

    with open(f'{model.path}/model.pkl', 'rb') as f:
        logistic_reg_model = pickle.load(f)
        
    X_test = np.load(f'{npy_data.path}/X_test.npy',allow_pickle=True)
    y_pred = logistic_reg_model.predict(X_test)

    if not os.path.exists(y_pred_file.path):
        os.makedirs(y_pred_file.path)

    np.save(f'{y_pred_file.path}/y_pred.npy', y_pred)
