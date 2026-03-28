output "instance_ip" {
  value       = oci_core_instance.inference.public_ip
  description = "Public IP address of the inference instance."
}

output "api_endpoint" {
  value       = "http://${oci_core_instance.inference.public_ip}:${var.api_port}/v1"
  description = "OpenAI-compatible API endpoint."
}

output "ssh_command" {
  value       = "ssh opc@${oci_core_instance.inference.public_ip}"
  description = "SSH command to connect to the instance."
}

output "model_deployed" {
  value       = local.model_id
  description = "Model ID deployed on the instance."
}

output "instance_type" {
  value       = var.instance_type
  description = "Instance type (gpu or cpu)."
}
