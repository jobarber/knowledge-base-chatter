import os
import io
import boto3
import json
import csv

# set environment variable with key ENDPOINT_NAME and value
# such as pytorch-inference-2021-07-08-22-02-27-487 in
# lambda function configuration
# for the api gateway, see the following url as an example:
# https://developers.facebook.com/blog/post/2020/08/03/connecting-web-app-pytorch-model-using-amazon-sagemaker

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))

    # 1
    data = json.loads(json.dumps(event))
    # print(data)
    payload = data['queryStringParameters']
    # print("PAYLOAD: ", payload)

    # 2
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=json.dumps(payload))  # xe
    # print(response)
    result = json.loads(response['Body'].read().decode())  # // 3
    # print(result)

    return result
