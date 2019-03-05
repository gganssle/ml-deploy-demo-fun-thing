#!/bin/bash

addy=$(aws ec2 describe-instances --instance-ids i-0a8f96036c177c242 --query 'Reservations[*].Instances[*].PublicIpAddress' --output text)

addy="$addy.us-west-2.compute.amazonaws.com"

echo $addy
