locals {
  model_id = var.model_selection == "custom" ? var.custom_model_url : var.model_selection
  shape    = var.instance_type == "gpu" ? "VM.GPU.A10.1" : "VM.Standard.E5.Flex"
}

# --- Data Sources ---

data "oci_identity_availability_domain" "ad" {
  compartment_id = var.compartment_ocid
  ad_number      = var.availability_domain_number
}

data "oci_core_images" "ol8" {
  compartment_id           = var.compartment_ocid
  operating_system         = "Oracle Linux"
  operating_system_version = "8"
  shape                    = local.shape
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

# --- Network ---

resource "oci_core_vcn" "main" {
  compartment_id = var.compartment_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = "turboinference-vcn"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "turboinference-igw"
  enabled        = true
}

resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "turboinference-rt"

  route_rules {
    network_entity_id = oci_core_internet_gateway.igw.id
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
  }
}

resource "oci_core_security_list" "main" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "turboinference-sl"

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }

  ingress_security_rules {
    protocol = "6" # TCP
    source   = var.api_allowed_cidr

    tcp_options {
      min = 22
      max = 22
    }
  }

  ingress_security_rules {
    protocol = "6" # TCP
    source   = var.api_allowed_cidr

    tcp_options {
      min = var.api_port
      max = var.api_port
    }
  }

  dynamic "ingress_security_rules" {
    for_each = var.enable_api_auth ? [1] : []
    content {
      protocol = "6"
      source   = var.api_allowed_cidr
      tcp_options {
        min = 8443
        max = 8443
      }
    }
  }
}

resource "oci_core_subnet" "public" {
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.main.id
  cidr_block        = "10.0.1.0/24"
  display_name      = "turboinference-subnet"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.main.id]
}

# --- Compute ---

resource "oci_core_instance" "inference" {
  compartment_id      = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domain.ad.name
  display_name        = "turboinference"
  shape               = local.shape

  dynamic "shape_config" {
    for_each = var.instance_type == "cpu" ? [1] : []
    content {
      ocpus         = var.cpu_ocpus
      memory_in_gbs = var.cpu_ram_gb
    }
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ol8.images[0].id
    boot_volume_size_in_gbs = var.boot_volume_gb
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public.id
    assign_public_ip = true
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(templatefile("${path.module}/cloud-init.yaml", {
      model_id        = local.model_id
      api_port        = var.api_port
      install_pi      = var.install_pi_agent
      enable_api_auth = var.enable_api_auth
    }))
  }
}
