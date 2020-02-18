import yaml
import os
import pandas as pd

from azureml.core import (
    Datastore,
    Dataset,
    RunConfiguration,
    Experiment,
    Workspace,
    ComputeTarget,
)

from azureml.exceptions import ComputeTargetException
from azureml.core.compute import AmlCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from sklearn.datasets import load_diabetes

# Load yaml and store it as a dictionary
with open("variables.yml", "r") as f:
    yaml_loaded = yaml.safe_load(f)['variables']

variables = {}
for d in yaml_loaded:
    variables[d['name']] = d['value']

# Authenticate with AzureML
#auth = ServicePrincipalAuthentication(
    #tenant_id=variables["tenant_id"],
    #service_principal_id=variables["service_principal_id"]
#)

aml_workspace = Workspace.get(
    subscription_id=variables["SUBSCRIPTION_ID"],
    resource_group = variables["RESOURCE_GROUP"],
    name = variables["BASE_NAME"]+"ws"
)


# Usually, the  cluster already exists, so we just fetch
try:
    compute_target = ComputeTarget(aml_workspace, variables["AML_COMPUTE_CLUSTER_CPU_SKU"])
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
    vm_size = variables['AML_COMPUTE_CLUSTER_SIZE'],
    vm_priority = variables['AML_CLUSTER_PRIORITY'],
    min_nodes = variables['AML_CLUSTER_MIN_NODES'],
    max_nodes = variables['AML_CLUSTER_MAX_NODES'],
    idle_seconds_before_scaledown = "300"
    )
    cpu_cluster = ComputeTarget.create(aml_workspace, variables["AML_COMPUTE_CLUSTER_CPU_SKU"], compute_config)

run_config = RunConfiguration(conda_dependencies=CondaDependencies.create(
        conda_packages=['numpy', 'pandas',
                        'scikit-learn', 'tensorflow', 'keras'],
        pip_packages=['azure', 'azureml-core',
                      'azure-storage',
                      'azure-storage-blob',
                      'azureml-dataprep'])
    )

# Pipeline definition

#Dataset creation
dataset_name = variables["DATASET_NAME"]

#Check to see if dataset exists
if (dataset_name not in aml_workspace.datasets):
    #Create dataset from diabetes sample data
    sample_data = load_diabetes()
    df = pd.DataFrame(
        data = sample_data.data,
        columns = sample_data.feature_names)
    df['Y'] = sample_data.target
    file_name = 'diabetes.csv'
    df.to_csv(file_name, index = False)

    #Upload file to default datastore in workspace

    default_ds = aml_workspace.get_default_datastore()
    target_path = 'training-data/'
    default_ds.upload_files(
        files = [file_name],
        target_path = target_path,
        overwrite = True,
        show_progress = False
    )

    # Register dataset

    path_on_datastore = os.path.join(target_path, file_name)
    dataset = Dataset.Tabular.from_delimited_files(
        path = (default_ds, path_on_datastore))
    dataset = dataset.register(
        workspace = aml_workspace,
        name = dataset_name,
        description = 'diabetes training data',
        tags = {'format': 'CSV'},
        create_new_version = True
    )

#Get the dataset
dataset = Dataset.get_by_name(aml_workspace, dataset_name)


#Create a PipelineData to pass data between steps
pipeline_data = PipelineData(
    'pipeline_data',
    datastore = aml_workspace.get_default_datastore()
)


train_model = PythonScriptStep(
    name = "Train Model",
    script_name = variables["TRAIN_SCRIPT_PATH"],
    compute_target = compute_target,
    run_config = run_config,
    inputs = [dataset.as_named_input('training_data')],
    outputs = [pipeline_data],
    allow_reuse = False,
    arguments = [
        "--step_output", pipeline_data
    ]
)

pipeline = Pipeline(
    workspace=aml_workspace,
    steps=[train_model],
    description="Builds sci_kit_learn model for diabetes",
)

if __name__ == "__main__":
    Experiment(aml_workspace, "fit-component-defects-model").submit(pipeline).wait_for_completion(
        show_output=True
    )
