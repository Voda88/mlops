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

with open("azure-variables.yaml", "r") as f:
    conf = yaml.safe_load(f, Loader=yaml.FullLoader)
    variables = conf["variables"]

# Authenticate with AzureML
#auth = ServicePrincipalAuthentication(
    #tenant_id=variables["tenant_id"],
    #service_principal_id=variables["service_principal_id"]
)

ws = Workspace(
    subscription_id=variables["subscription_id"],
    resource_group=variables["resource_group"],
    workspace_name=variables["workspace_name"],   
)

# Usually, the  cluster already exists, so we just fetch
try:
    compute_target = ComputeTarget(ws, variables["AML_COMPUTE_CLUSTER_CPU_SKU"])
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size = variables['AML_COMPUTE_CLUSTER_SIZE'], max_nodes = variables['AML_CLUSTER_MAX_NODES'])

run_config = RunConfiguration(
    conda_dependencies=CondaDependencies(
        conda_dependencies_file_path="./environment.yaml"
    )
)

# Pipeline definition
inputdata = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name="data",
    container_name="component-condition-model",
    account_name="topsecretdata",
    account_key=os.environ["ACCOUNT_KEY"],
)

train_model = PythonScriptStep(
    script_name="./train.py",
    name="fit-nlp-model",
    inputs=[inputdata.as_download()],
    runconfig=run_config,
    compute_target=compute_target,
)

pipeline = Pipeline(
    workspace=ws,
    steps=[train_model],
    description="Builds Keras model for detecting component defects",
)

if __name__ == "__main__":
    Experiment(ws, "fit-component-defects-model").submit(pipeline).wait_for_completion(
        show_output=True
    )
