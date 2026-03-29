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

output "api_endpoint_auth" {
  description = "Authenticated API endpoint (if auth enabled)."
  value       = var.enable_api_auth ? "http://${oci_core_instance.inference.public_ip}:8443/v1 (requires API key from /opt/turboinference/api-key)" : "N/A (auth disabled)"
}
