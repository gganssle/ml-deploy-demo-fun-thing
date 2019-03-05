#!/bin/bash

# fire up deep learning instance
instanceid=$(aws ec2 run-instances \
  --image-id ami-0027dfad6168539c7 \
  --count 1 \
  --instance-type t2.micro \
  --key-name ml-demo-aws-keys \
  --security-group-ids sg-0112f0a730fbbeecf \
  --subnet-id subnet-04c01aa3e5afd9043 \
  --associate-public-ip-address \
  | grep InstanceId \
  | tail -c 22 \
  | head -c -3)


echo "instance id = $instanceid"

sleep 120

addy=$(aws ec2 describe-instances --instance-ids $instanceid --query 'Reservations[*].Instances[*].PublicIpAddress' --output text)

addy=${addy//./-}

addy="ec2-$addy.us-west-2.compute.amazonaws.com"

echo "instance IP address = $addy"

# initialize deep learning environment
ssh -oStrictHostKeyChecking=no -i ~/.ssh/ml-demo-aws-keys ubuntu@$addy << HERE
source activate tensorflow_p36
sleep 120
mkdir dat
mkdir model
HERE

# do work on the instance
rsync -avz --progress -e 'ssh -i ~/.ssh/ml-demo-aws-keys' ../../model/tf_linear.py ubuntu@$addy:~/model/
rsync -avz --progress -e 'ssh -i ~/.ssh/ml-demo-aws-keys' ../../dat/clean.csv ubuntu@$addy:~/dat/

ssh -i ~/.ssh/ml-demo-aws-keys ubuntu@$addy << HERE
source activate tensorflow_p36
cd model
time python tf_linear.py --model-dir checkpoints/
HERE

# kill off instance
echo "killing instance now"

aws ec2 stop-instances \
  --instance-ids $instanceid

echo "done"
