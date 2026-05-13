## Step 1: Download clouds.yaml from Horizon

From: Project → API Access → Download OpenStack clouds.yaml

## Step 2: Move clouds.yaml to correct CLI location

On macOS, OpenStack expects configuration in: ~/.config/openstack/clouds.yaml

follow the commands below:
Created directory: mkdir -p ~/.config/openstack

Moved file:
mv ~/Downloads/clouds.yaml ~/.config/openstack/clouds.yaml
chmod 600 ~/.config/openstack/clouds.yaml

## Step 3: Create Application Credential

From Horizon: Project → Identity → Application Credentials → Create

## Step 4: Update clouds.yaml

ensure following configuration (auth url must be in v3):

clouds:
  openstack:
    auth:
      auth_url: https://pegasus.sky.oslomet.no:5000/v3
      application_credential_id: "<ID>"
      application_credential_secret: "<SECRET>"
    region_name: "Pilestredet"
    interface: "public"
    identity_api_version: 3
    auth_type: "v3applicationcredential"


## Step 5: Validation

openstack --os-cloud openstack server list

## Step 6: To avoid —os-cloud

run:  export OS_CLOUD=openstack

then, check: openstack server list