from kfp import compiler
from kfp.dsl import pipeline
from components import prepare_data, train_test_split, training_basic_classifier, predict_on_test_data

@pipeline(
    name="iris-pipeline",
)
def iris_pipeline():

    # Data preparation
    prepare_data_task = prepare_data()
    
    # Split data into Train and Test set
    train_test_split_task = train_test_split(df_data = prepare_data_task.outputs["df_data"]) 
    train_test_split_task.after(prepare_data_task)
    
    # Model training
    training_basic_classifier_task = training_basic_classifier(npy_data = train_test_split_task.outputs["npy_data"])
    training_basic_classifier_task.after(train_test_split_task)
    
    # Model evaluation
    predict_on_test_data_task = predict_on_test_data(model = training_basic_classifier_task.outputs["model"], npy_data = train_test_split_task.outputs["npy_data"])
    predict_on_test_data_task.after(training_basic_classifier_task)
    

if __name__ == '__main__':
    compiler.Compiler().compile(iris_pipeline, 'iris_pipeline.yaml')