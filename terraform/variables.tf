variable "compartment_ocid" {
  type        = string
  description = "OCID of the compartment to deploy resources into."
}

variable "region" {
  type        = string
  default     = "us-ashburn-1"
  description = "OCI region for deployment."
}

variable "availability_domain_number" {
  type        = number
  default     = 1
  description = "Availability domain number (1-based index)."
}

variable "model_selection" {
  type        = string
  default     = "Qwen/Qwen3.5-35B-A3B"
  description = "HuggingFace model ID to deploy, or 'custom' to provide your own URL."
}

variable "custom_model_url" {
  type        = string
  default     = ""
  description = "Custom model URL or HuggingFace ID. Only used when model_selection is 'custom'."
}

variable "instance_type" {
  type        = string
  default     = "gpu"
  description = "Instance type: 'gpu' for VM.GPU.A10.1, 'cpu' for VM.Standard.E5.Flex."

  validation {
    condition     = contains(["gpu", "cpu"], var.instance_type)
    error_message = "instance_type must be 'gpu' or 'cpu'."
  }
}

variable "cpu_ocpus" {
  type        = number
  default     = 16
  description = "Number of OCPUs for flex CPU instances."
}

variable "cpu_ram_gb" {
  type        = number
  default     = 128
  description = "RAM in GB for flex CPU instances."
}

variable "ssh_public_key" {
  type        = string
  description = "SSH public key for instance access."
}

variable "api_allowed_cidr" {
  type        = string
  default     = "0.0.0.0/0"
  description = "CIDR block allowed to access the API port."
}

variable "install_pi_agent" {
  type        = bool
  default     = true
  description = "Whether to install the Pi coding agent."
}

variable "api_port" {
  type        = number
  default     = 8080
  description = "Port for the inference API."
}

variable "boot_volume_gb" {
  type        = number
  default     = 200
  description = "Boot volume size in GB."
}
