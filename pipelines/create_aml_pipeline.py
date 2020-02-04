import yaml
import os

from azureml.core import (
    Datastore,
    RunConfiguration,
    Experiment,
    Workspace,
    ComputeTarget,
)

from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

print(os.getcwd())

with open("variables.yaml", "r") as f:
    conf = yaml.safe_load(f, Loader=yaml.FullLoader)
    variables = conf["variables"]

# Authenticate with AzureML
#auth = ServicePrincipalAuthentication(
    #tenant_id=variables["tenant_id"],
    #service_principal_id=variables["service_principal_id"]
#)

ws = Workspace(
    subscription_id=variables["SUBSCRIPTION_ID"],
    resource_group=variables["RESOURCE_GROUP"],
    workspace_name=variables["BASE_NAME"]+"ws"   
)


# Usually, the  cluster already exists, so we just fetch
try:
    compute_target = ComputeTarget(ws, variables["AML_COMPUTE_CLUSTER_CPU_SKU"])
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = Amlcompute.provisioning_configuration(
    vm_size = variables['AML_COMPUTE_CLUSTER_SIZE'],
    vm_priority = variables['AML_CLUSTER_PRIORITY'],
    min_nodes = variables['AML_CLUSTER_MIN_NODES'],
    max_nodes = variables['AML_CLUSTER_MAX_NODES'],
    idle_seconds_before_scaledown = "300"
    )
    cpu_cluster = ComputeTarget.create(ws, variables["AML_COMPUTE_CLUSTER_CPU_SKU"], compute_config)

run_config = RunConfiguration(
    conda_dependencies= CondaDependencies(
        conda_dependencies_file_path="./conda_dependencies.yml"
    )
)

# Pipeline definition
dataset_name = variables["DATASET_NAME"]
datastore = Datastore.get(ws, variables["DATASTORE_NAME"])
data_path = [(datastore, variables["DATAFILE_PATH"])]
dataset = Dataset.Tabular.from_delimited_files(path = data_path)
dataset.register(workspace = ws,
                name = dataset_name,
                description = "dataset with training data",
                create_new_version = True)

train_model = PythonScriptStep(
    name = "Train Model",
    script_name = variables["TRAIN_SCRIPT_PATH"],
    compute_target = compute_target,
    run_config = run_config,
    inputs = [dataset]
)

pipeline = Pipeline(
    workspace=ws,
    steps=[train_model],
    description="Builds sci_kit_learn model for diabetes",
)

if __name__ == "__main__":
    Experiment(ws, "fit-component-defects-model").submit(pipeline).wait_for_completion(
        show_output=True
    )
