import yaml
import os
import pandas as pd
import argparse

from azureml.core import (
    Datastore,
    Dataset,
    RunConfiguration,
    Experiment,
    Workspace,
    ComputeTarget,
    Environment
)

from azureml.exceptions import ComputeTargetException
from azureml.core.compute import AmlCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from sklearn.datasets import load_diabetes

#Create parser to import connection parameters as arguments
parser = argparse.ArgumentParser("aml_pipeline")
parser.add_argument(
    "--subscription_id",
    type = str,
    help = "ID of subscription where AML workspace is. Defined in DevOps variable group Connection_parameters SUBSCRIPTION_ID."
)
parser.add_argument(
    "--resource_group",
    type = str,
    help = "Name of resource group where AML workspace is. Defined in DevOps variable group Connection_parameters RESOURCE_GROUP."
)
parser.add_argument(
    "--base_name",
    type = str,
    help = "Base name of Azure resources. Defined in DevOps variable group Connection_parameters BASE_NAME."
)
args = parser.parse_args()

#Find workspace using connection parameters
aml_workspace = Workspace.get(
    subscription_id= args.subscription_id,
    resource_group = args.resource_group,
    name = args.base_name+"ws"
)

# Load yaml and store it as a dictionary
with open("variables.yml", "r") as f:
    yaml_loaded = yaml.safe_load(f)['variables']

variables = {}
for d in yaml_loaded:
    variables[d['name']] = d['value']


# Check if compute cluster exists. If not, create one.
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

#create environment from conda_dependencies.yml for runconfig
environment = Environment(name="myenv")
conda_dep = CondaDependencies(conda_dependencies_file_path = "conda_dependencies.yml")
environment.python.conda_dependencies = conda_dep
run_config = RunConfiguration()
run_config.environment = environment

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


#Configure step for training model
train_model = PythonScriptStep(
    name = "Train Model",
    script_name = variables["TRAIN_SCRIPT_PATH"],
    compute_target = compute_target,
    runconfig = run_config,
    inputs = [dataset.as_named_input('training_data')],
    outputs = [pipeline_data],
    allow_reuse = False,
    arguments = [
        "--step_output", pipeline_data
    ]
)

#Configure step for evaluating model


#Configure pipeline
pipeline = Pipeline(
    workspace=aml_workspace,
    steps=[train_model],
    description="Builds sci_kit_learn model for diabetes",
)

if __name__ == "__main__":
    Experiment(aml_workspace, "diabetes_model").submit(pipeline).wait_for_completion(
        show_output=True
    )
