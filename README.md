# MWPA Thesis Implementation

This repository contains the implementation of my master thesis:

**MWPA: A Modular Framework from ML-Based Multivariate Workload Prediction to Price-Aware Cloud Provider Selection via VM State Actuation in Multi-Cloud**

MWPA is a lightweight modular framework that connects:
- ML-based multivariate workload forecasting
- cloud pricing input
- a rule-based decision engine
The framework performs provider switching through VM-state actuation, where the selected provider VM remains active while the non-selected provider VM is deactivated. So, the goal of the framework is to support multivariate workload forecast and cost-aware actuation for cloud provider selection in multi-cloud environments at infrastructure level using virtual machines that improve cost by switching providers.

## Repository Scope

This repository includes the main implementation code, configuration files, selected processed outputs, plots, and experiment artifacts used to develop and validate the MWPA framework.

The repository covers:
- Alibaba workload data preprocessing and multivariate time-series dataset construction
- forecasting model training and evaluation
- pricing and cost-estimation logic
- decision-engine implementation
- OpenStack-based cloud experiments
- selected real multi-cloud experiment outputs
- cost improvement analysis using MWPA

## Repository Structure

- `config/`  
  YAML configuration files for preprocessing, pricing, and decision-engine settings.

- `scripts/`  
  Main scripts for model training, inference, pricing, cost calculation, and decision-engine execution.

- `scripts/Datasetup/`  
  Scripts for preprocessing Alibaba workload traces and generating intermediate and final time-series datasets.

- `scripts/openstack_experiments/`  
  Scripts and notes related to OpenStack-based validation and live cloud experiments.

- `output/`  
  Selected processed datasets, trained-model outputs, forecast results, prices, and decision logs.

- `plots/`  
  Selected visualizations used for analysis and thesis figures.

## Main Workflow

The overall MWPA workflow is implemented in the following stages:

1. Preprocess Alibaba cluster trace data to aggregate CPU, memory, and request-rate metrics into a multivariate time-series dataset.
2. Train and evaluate the forecasting model. Also, save the trained model and scaler for later inference.
3. Prepare cloud pricing and decision engine input.
4. Run the decision engine in static or live mode. 
    - It estimates provider cost from forecasted demand. 
    - And it then applies VM-state actuation based on the selected providers. 
    - Finally, it store decision logs and selected experiment outputs.

## How to Reproduce the Experiments

This repository can be used to reproduce major parts of the MWPA workflow, including preprocessing, forecasting, pricing input preparation, and decision-engine validation. Some cloud experiments can also be repeated if the required infrastructure is prepared.

### Prerequisites

Before running the experiments, the user should prepare:

- access to the Alibaba Cluster Trace 2021 dataset
- a Python environment with the required packages installed
- OpenStack access for VM-based experiments
- AWS access if reproducing the real multi-cloud validation
- valid API credentials, SSH keys, and local CLI configuration
- equivalent VM names, instance identifiers, paths, and pricing settings updated in the configuration files

### Recommended Execution Order

To reproduce the workflow, the recommended order is:

1. Run the preprocessing scripts in `scripts/Datasetup/` to prepare service-based and time-based datasets from the Alibaba traces.
2. Generate the final multivariate time-series dataset used for training.
3. Run the forecasting model training and evaluation scripts.
4. Save and transfer the trained model artifacts and scaler if cloud-side inference is required.
5. Prepare provider pricing input using the configured pricing mode.
6. Run the decision engine in static mode for controlled validation.
7. Run the decision engine in live mode for remote inference and VM-state switching experiments.
8. Review the generated forecast files, decision logs, and cost analysis outputs.


## Reusing the Cloud Experiments

To reuse the OpenStack and real multi-cloud experiments, users must prepare equivalent cloud environments and update the configuration files according to their own setup.

This includes:
- VM instance names and identifiers
- remote SSH settings and paths
- OpenStack and AWS CLI authentication
- forecast file paths
- pricing mode and provider pricing values

The implementation assumes that application environments are already prepared on the candidate provider VMs. The framework performs provider selection through VM-state actuation rather than full application deployment.

## Experiment Context

The repository supports three main stages of validation:

- **Offline stage:** preprocessing, time-series construction, model training, and forecasting evaluation
- **OpenStack stage:** transferred-model validation, live telemetry collection, and decision-engine execution in an emulated multi-cloud testbed
- **Real multi-cloud stage:** decision-engine validation using OpenStack and AWS

## Notes

The following items are not included in this repository:
- raw dataset zip files
- virtual environments
- Python cache files
- private credentials
- API secrets
- SSH keys
- cloud account-specific identifiers

Some outputs are included in selected form for documentation and reproducibility, but the repository does not contain all raw experiment files.

## Important Reproducibility Note

This repository does not provide full plug-and-play reproducibility. Some parts of the experiments depend on external datasets, private credentials, cloud infrastructure, and environment-specific configuration that are not included here. However, the repository contains the main implementation logic and selected outputs needed to understand, reuse, and adapt the MWPA workflow.