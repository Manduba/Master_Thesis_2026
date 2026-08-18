# PACE Implementation Workflow

This document provides a script-level overview of the implementation workflow used in the PACE framework. It complements the main `README.md` by showing how the main modules, configuration files, and experiment stages are connected.

The workflow is divided into four parts:

1. dataset setup and preprocessing  
2. model training and evaluation  
3. pricing and decision logic  
4. cloud-side experiments and validation  

---

## 1. Dataset Setup and Preprocessing

This stage prepares the Alibaba Cluster Trace 2021 data into the multivariate time-series dataset used for model training.

### 1.1 Raw data preparation
In the script folder, a subfolder contains all the process from data download to preprocesssing.

**Scripts**
- `raw_data_dl.py`
- `raw_data_inspect.py`

**Main Purpose**
- download or organize the raw Alibaba trace files
- inspect the structure of the raw resource and request-rate files
- checked raw files for preprocessing
- verified column structure and time spans

---

### 1.2 Resource metrics preprocessing

**Script**
- `scripts/Datasetup/resample_resource.py`

**Config**
- `config/resource.yaml`

**Purpose**
- process raw `MSResource_Table` files
- convert timestamps into fixed time windows
- compute service-level CPU and memory features 

**Main outputs**
- CPU and memory mean
- CPU and memory standard deviation
- active instance counts

---

### 1.3 Request-rate metrics preprocessing

**Script**
- `scripts/Datasetup/resample_msrtqps.py`

**Config**
- `config/msrtqps.yaml`

**Purpose**
- process raw `MS_MCR_RT_Table` files
- aggregate request-rate related metrics
- compute service-level request-rate features

**Main outputs**
- total request rate
- request-rate mean
- request-rate standard deviation
- latency-related intermediate values

---

### 1.4 Merge service-level metrics

**Script**
- `scripts/Datasetup/merge_metrics.py`

**Config**
- `config/merge.yaml`

**Purpose**
- merge resource and request-rate outputs using common time and service keys
- produce a unified service-level dataset

**Main output**
- merged service-level multivariate dataset

---

### 1.5 Time-based aggregation

**Script**
- `scripts/Datasetup/clusterlevel_timebased.py`

**Config**
- `config/cluster_timebased.yaml`

**Purpose**
- aggregate service-level metrics into one cluster-level time series
- generate average CPU, memory, and request-rate values over time

**Main output**
- batch-wise time-based dataset

---

### 1.6 Join batches into final dataset

**Script**
- `scripts/Datasetup/join_cluster_service_batch.py`

**Config**
- `config/join_cluster_service_batch.yaml`

**Purpose**
- combine all processed batches into the final dataset used for forecasting

**Main output**
- `output/final_service_timebased.csv`

---

### 1.7 Exploratory time-series analysis

**Script**
- `scripts/Datasetup/visualization_clean_data.py`

**Purpose**
- inspect the final time-series dataset visually
- generate plots for CPU, memory, and request-rate behavior

**Main outputs**
- plots for exploratory analysis
- selected figures used in the thesis

---

## 2. Model Training and Evaluation

This stage trains the forecasting model and evaluates prediction performance.

### 2.1 Model training

**Script**
- `scripts/prediction_hf.py`

**Purpose**
- load the final multivariate time-series dataset (final_service_timebased.csv)
- apply MinMax normalization
- train the Hugging Face TimeSeriesTransformer model
- save the trained model and scaler

**Main outputs**
- trained model artifacts
- saved scaler
- forecast results
- evaluation metrics
- training loss curve

---

### 2.2 Post-training evaluation and plots

**Purpose**
- evaluate CPU, memory, and request-rate prediction
- generate performance metrics and visual comparison plots

**Main outputs**
- MAE, RMSE, sMAPE values
- true vs predicted plots
- selected evaluation figures for the thesis

---

## 3. Pricing and Decision Logic

This stage prepares provider pricing and runs the decision engine.

### 3.1 Price fetching

**Script**
- `scripts/fetch_prices.py`

**Purpose**
- fetch provider-specific VM prices
- support AWS live pricing and other prepared pricing inputs

**Main outputs**
- price data in usable format
- optional CSV-based pricing output

---

### 3.2 Cost calculation

**Script**
- `scripts/cost_module.py`

**Purpose**
- load pricing information
- support live, CSV, and simulated pricing modes
- prepare provider cost values for the decision engine

**Main outputs**
- comparable provider cost inputs

---

### 3.3 Decision engine

**Script**
- `scripts/decision_engine_main.py`
- `scripts/decision_engine_mock.py`

**Config**
- `config/decision.yaml`

**Purpose**
- read forecast output and pricing input
- estimate required VM units
- compare provider cost
- select the cheaper provider
- apply VM-state actuation
- write decision logs

**Modes**
- static mode
- live mode

**Main outputs**
- decision logs
- selected provider state changes
- forecast-driven VM switching results in real multi-cloud / testbed

---

## 4. OpenStack and Multi-Cloud Experiments

This stage validates PACE in cloud environments.

### 4.1 Model transfer to OpenStack VM

**Script**
- `inference_next10min.py`

**Purpose**
- load the trained model and scaler in the OpenStack VM
- run cloud-side inference using prepared input data
- verify that the forecasting component works after transfer

**Main outputs**
- forecast CSV inside the VM
- deployment validation output

---

### 4.2 Controlled workload experiment in OpenStack

**Scripts_inside_VM**
- install inginx in the vm where model is transferred.
- `run_experiment_http.sh`( Run if for 75 minutes if you are first time user)
- otherwise download "openstack_metrics_http_75min.csv" from the output folder and transfer it the vm. The shell script will append metrics in this file, then run the inference script.
- `log_openstack_http_metrics.py` (do not have to run this file. This combined inside the shell script to generate metrics file)
- `infer_openstack_http_next10min.py`

**Purpose**
- generate live workload using Nginx and Siege
- log CPU, memory, and request-rate telemetry
- run inference on live cloud-generated input

**Main outputs**
- live telemetry CSV
- forecast CSV from live workload
- validation outputs for cloud-side forecasting

---

### 4.3 Decision engine execution in OpenStack testbed

--- For exceuting this 3 vms are needed. one where the model is transferred and other for candidate provoder reprenstetives.

**Inside VM**
- `create two vm in openstack (chosen platform) with configuration  2GB RAM, 2vCPUs and 50GB.
-  name one Cloud A-Op and another Cloud B-aws
- deploy an app (ex.sockshop to make the candidate environment ready)
- `run_live_demo.sh` (run it for  70 minutes if you are first time user. you can keep the model anywhere whether remote vm or local machine)
- otherwise download "openstack_metrics_http_75min_live.csv" from the output folder and transfer it the vm here model is tranferred. The shell script will append metrics in this file, then run the inference script.
- `infer_openstack_http_live.py` (transfer it to the vm but do not run)
- `log_openstack_http_live_metrics.py` (transfer it to the vm but do not run)
- `./stop_live_demo.sh` stop the logger when needed

**Local/controller side**
- `decision_enigne_mock.py`
- `config/decision.yaml` (changed based on you prefrene for static/live mode, sleep time etc.)
- for static mode check: download "predictions_http_next10min.csv" from the output folder. and then run the decicion_engine_mock.py

**Purpose**
- validate the decision engine in an emulated multi-cloud OpenStack setup
- test both static and live decision paths
- verify VM switching behavior using provider-representative VMs

**Main outputs**
- OpenStack decision logs
- VM-state switching records

---

### 4.4 Decision engine execution in real multi-cloud

--- For exceuting this 3 vms are needed. one where the model is transferred and other for candidate provoder reprenstetive.

**Inside VM**
- create one vm in openstack and another in aws with configuration  2GB RAM, 2vCPUs and 50GB. (you can choose your preferred cloud platform)
- chmod +x run_live_demo.sh (make the command executable)
- `./run_live_demo.sh` in the vm where the model is tranferred. (you can keep the model anywhere whether remote vm or local machine)
- otherwise download "openstack_metrics_http_75min_live.csv" from the output folder and transfer it the vm here model is tranferred. The shell script will append metrics in this file, then run the inference script.
- `infer_openstack_http_live.py` (do not run but should be inside the vm where model is tranferred)
- `log_openstack_http_live_metrics.py` (do not run but should be inside the vm where model is tranferred)

**Local/controller side**
- `decision_enigne_main.py`
- `config/decision.yaml` (change static/live mode based on you preference)

**Purpose**
- validate the decision engine in a real multi-cloud setup using OpenStack and AWS
- reuse the same core logic with provider-specific actuation updates

**Main outputs**
- real multi-cloud decision logs
- AWS/OpenStack switching records
- cost analysis input data

## 5. Cost improve analysis 

**Script**
- `cost_improve_analysis_real_multicloud.py`

**Purpose**
- Valiadte cost optimization improvement using PACE using decision log from real multi cloud excecution.

**Main outputs**
- comparisons of PACE vs always AWS using
- comparisons of PACE vs always Openstack using
- Quantitative analysis of the output results

---

## 5. Suggested Usage Paths

Depending on the goal, the repository can be used in different ways.

### 6.1 Full workflow reproduction
Use this path if the goal is to reproduce the entire PACE implementation:
1. dataset setup and preprocessing
2. model training and evaluation
3. pricing preparation
4. decision-engine execution
5. cloud experiment validation

### 6.2 Decision-engine-only reuse
Use this path if trained model artifacts, scaler, and cloud environments are already prepared:
1. verify model and scaler paths
2. update `decision.yaml`
3. prepare pricing input
4. run decision engine in static or live mode
5. review decision logs and outputs

### 6.3 Cloud-experiment-only reuse
Use this path if the goal is only to validate inference and switching behavior in OpenStack or real multi-cloud:
1. prepare cloud VMs and authentication
2. transfer model and scaler
3. run workload generation and inference
4. execute decision engine
5. review generated logs and forecast outputs

---

## 7. Notes

- Some script names and paths are environment-specific and may need adjustment.
- Raw datasets, credentials, SSH keys, and cloud account-specific identifiers are not included in the repository.
- The framework assumes that application environments are already prepared on the candidate provider VMs.
- The implementation focuses on VM-state actuation, not full application redeployment or orchestration.
