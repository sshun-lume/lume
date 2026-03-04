import os
import platform
import subprocess
import psutil
import requests
import mlflow

class MLflowEnvLogger:
    @staticmethod
    def get_cpu_model():
        """OSごとの標準コマンドでCPUモデル名を取得"""
        os_type = platform.system()
        try:
            if os_type == "Linux":
                cmd = "grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2"
                return subprocess.check_output(cmd, shell=True).decode().strip()
            elif os_type == "Windows":
                cmd = "wmic cpu get name /value"
                out = subprocess.check_output(cmd, shell=True).decode()
                return out.split('=')[-1].strip()
            elif os_type == "Darwin":
                cmd = "sysctl -n machdep.cpu.brand_string"
                return subprocess.check_output(cmd, shell=True).decode().strip()
        except Exception:
            return platform.processor()
        return "Unknown"

    @staticmethod
    def get_cuda_info():
        """nvidia-smiからCUDAとドライバ情報を取得"""
        info = {"env.cuda_version": "N/A", "env.gpu_driver": "N/A"}
        try:
            # CUDA Versionの抽出
            out = subprocess.check_output("nvidia-smi", shell=True, encoding="utf-8")
            for line in out.splitlines():
                if "CUDA Version" in line:
                    parts = line.split("CUDA Version:")[1].split("|")[0].strip()
                    info["env.cuda_version"] = parts
                if "Driver Version" in line:
                    parts = line.split("Driver Version:")[1].split("CUDA Version:")[0].strip()
                    info["env.gpu_driver"] = parts
        except Exception:
            pass
        return info

    @staticmethod
    def get_cloud_instance_type():
        """クラウドのメタデータサービスからインスタンスタイプを取得"""
        # AWS (IMDSv2)
        try:
            token = requests.put("http://169.254.169.254", 
                                 headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}, timeout=1).text
            return "aws", requests.get("http://169.254.169.254", 
                                       headers={"X-aws-ec2-metadata-token": token}, timeout=1).text
        except: pass

        # GCP
        try:
            resp = requests.get("http://metadata.google.internal", 
                                headers={"Metadata-Flavor": "Google"}, timeout=1)
            return "gcp", resp.text.split('/')[-1]
        except: pass

        # Azure
        try:
            resp = requests.get("http://169.254.169.254", 
                                headers={"Metadata": "true"}, timeout=1)
            return "azure", resp.text
        except: pass

        return "local", "personal-pc"

    @classmethod
    def log_all_env_tags(cls):
        """すべての固定環境情報をMLflowタグとして記録"""
        cloud_provider, instance_type = cls.get_cloud_instance_type()
        
        tags = {
            # CPU & Memory
            "env.cpu_model": cls.get_cpu_model(),
            "env.cpu_cores_physical": psutil.cpu_count(logical=False),
            "env.cpu_max_freq_mhz": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
            "env.mem_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            
            # OS & Runtime
            "env.os": f"{platform.system()} {platform.release()}",
            "env.python_version": platform.python_version(),
            
            # Cloud
            "env.cloud_provider": cloud_provider,
            "env.instance_type": instance_type,
        }
        
        # GPU/CUDA
        tags.update(cls.get_cuda_info())
        
        #mlflow.set_tags(tags)
        return tags
