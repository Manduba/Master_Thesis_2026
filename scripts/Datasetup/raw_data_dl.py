import os
import subprocess

def main():
    # Step 1: Git clone
    print("Cloning repository...")
    subprocess.run(["git", "clone", "https://github.com/alibaba/clusterdata"])
    os.chdir("clusterdata")
    
    # Step 2: Run fetchData.sh
    print("Running fetchData.sh...")
    subprocess.run(["bash", "fetchData.sh"])
    
    # Step 3: Download and extract MSResource files
    print("Processing MSResource files...")
    os.chdir("MSResource")
    for i in range(0, 1):
        url = f"http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSResource/MSResource_{i}.tar.gz"
        subprocess.run(["curl", "-O", url])
    
    for file in os.listdir('.'):
        if file.startswith("MSResource_") and file.endswith(".tar.gz"):
            subprocess.run(["tar", "-xzf", file])
            os.remove(file) #remove tar
    
    # Step 4: Download and extract MSRTQps files
    print("Processing MSRTQps files...")
    os.chdir("../MSRTQps")
    for i in range(0, 2):
        url = f"http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2021MicroservicesTraces/MSRTQps/MSRTQps_{i}.tar.gz"
        print(f"Downloading MSRTQps_{i}.tar.gz...")
        subprocess.run(["curl", "-O", url])
    
    for file in os.listdir('.'):
        if file.startswith("MSRTQps_") and file.endswith(".tar.gz"):
            subprocess.run(["tar", "-xzf", file])
            os.remove(file) #remove tar
    
    print("Download complete!")

if __name__ == "__main__":
    main()
