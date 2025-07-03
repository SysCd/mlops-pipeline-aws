provider "aws" {
  region = "eu-west-2" # London
}

resource "aws_vpc" "mlops_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
}

resource "aws_internet_gateway" "mlops_gw" {
  vpc_id = aws_vpc.mlops_vpc.id
}

resource "aws_subnet" "mlops_subnet" {
  vpc_id                  = aws_vpc.mlops_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
}

resource "aws_route_table" "mlops_route_table" {
  vpc_id = aws_vpc.mlops_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mlops_gw.id
  }
}

resource "aws_route_table_association" "mlops_rt_assoc" {
  subnet_id      = aws_subnet.mlops_subnet.id
  route_table_id = aws_route_table.mlops_route_table.id
}

resource "aws_security_group" "mlops_sg" {
  name        = "mlops_sg"
  description = "Allow SSH and FastAPI"
  vpc_id      = aws_vpc.mlops_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

 ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }


  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_key_pair" "mlops_key" {
  key_name   = "mlops-key-v2"
  public_key = file("${path.module}/keys/ML-Key.pub")
}





resource "aws_instance" "mlops_ec2" {
  ami                    = "ami-0f4f4482537714bd9"
  instance_type          = "t3.small"
  key_name               = aws_key_pair.mlops_key.key_name
  subnet_id              = aws_subnet.mlops_subnet.id
  vpc_security_group_ids = [aws_security_group.mlops_sg.id]
  associate_public_ip_address = true

user_data = <<-EOF
              #!/bin/bash
              yum update -y
              yum install -y docker
              systemctl start docker
              usermod -aG docker ec2-user
              EOF


  tags = {
    Name = "mlops-api-server"
  }
}

resource "aws_eip" "static" {
  instance = aws_instance.mlops_ec2.id
  
  tags = {
    Name = "mlops-static-ip"
  }
}



# Remote S3 backend to keep track of file changes
terraform {
  backend "s3" {
    bucket         = "tf-state-bucket-mlops-876512"
    key            = "env:/terraform.tfstate"
    region         = "eu-west-2"
  }
}
output "instance_public_ip" {
  value = aws_instance.mlops_ec2.public_ip
}

