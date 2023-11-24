import random
import sys, os, json, argparse
from datetime import datetime

import anyio
import dagger

async def main(args):
    config = dagger.Config(log_output=sys.stdout)

    async with dagger.Connection(config) as client:
        # Load Pulumi
        pulumi_access_token = client.set_secret("PULUMI_ACCESS_TOKEN", args.pulumi_token)
        pulumi_stack = args.stack
        aws_access_key_id = client.set_secret("AWS_ACCESS_KEY_ID", args.aws_id)
        aws_secret_access_key = client.set_secret("AWS_SECRET_ACCESS_KEY", args.aws_secret)
        aws_session_token = client.set_secret("AWS_SESSION_TOKEN", args.aws_token)

        pulumi_container = (
            client.container()
            .from_("pulumi/pulumi-python")
            .with_exec(["/bin/bash", "-c", "pulumi plugin install resource aws"])
            .with_secret_variable("PULUMI_ACCESS_TOKEN", pulumi_access_token)
            .with_directory(
                "/pulumi/projects",
                client.host().directory("ci/infra"),
                exclude=["ecr/venv", "ecs/venv", "ec2/venv"]
            )
            # Mount AWS credentials
            .with_secret_variable("AWS_ACCESS_KEY_ID", aws_access_key_id)
            .with_secret_variable("AWS_SECRET_ACCESS_KEY", aws_secret_access_key)
            .with_secret_variable("AWS_SESSION_TOKEN", aws_session_token)
        )

        # Create the ECR registry:
        ecr_output = await (
            pulumi_container
            .with_env_variable("CACHEBUSTER", str(datetime.now()))
            .with_exec(["/bin/bash", "-c", "cd ecr &&  pulumi stack select "+pulumi_stack+" -c && pulumi up -y"])
        )

        # Get ECR url
        repo_url = await (
            ecr_output
            .with_exec(["/bin/bash", "-c", "cd ecr && pulumi stack output repository_url -s "+pulumi_stack])
        ).stdout()

        # Get ECR token
        auth_token = await (
            ecr_output
            .with_exec(["/bin/bash", "-c", "cd ecr && pulumi stack output authorization_token"])
        ).stdout()

        # print(f"auth_token: {auth_token}")

        auth_token = json.loads(auth_token)
        password = client.set_secret("password", auth_token["password"])

        # Compiling the model for torchserve

        torchserve_img_dir = (
            client.host()
            .directory(".",
                       exclude=[
                           "checkpoint/",
                           "model_store/",
                           "data/",
                           "output/",
                           "ci/",
                           "output/",
                           "output_test/"
                       ],
                       # include=[
                       #     # "scripts/",
                       #     "model/",
                       #     "*py"
                       # ]
                       )
        )

        torchserve_container = (
            torchserve_img_dir
            # .with_workdir("/src/crypsis-delizziosa-model/")
            .docker_build(dockerfile="docker/Dockerfile", platform=dagger.Platform("linux/amd64"))
        )

        # Compile the model to .mar file
        # compile_model = await (
#             torchserve_container
#             .with_exec(["/bin/bash", "-c", "torch-model-archiver --model-name vToonify --version 1.0 \
# --serialized-file ./checkpoint/arcane/vtoonify_s_d.pt \
# --model-file ./model/vtoonify.py \
# --handler ./main \
# --export-path /home/model-server/model_store \
# --extra-files ./util.py,./model/vtoonify.py,./model/dualstylegan.py,./model/stylegan/stylegan_model.py,./model/stylegan/op/__init__.py,./model/stylegan/op/upfirdn2d_pkg.py,./model/stylegan/op/fused_act.py,./model/encoder/align_all_parallel.py,./model/bisenet/bisnet_model.py,./model/bisenet/resnet.py,./model/stylegan/op/upfirdn2d_kernel.cu,./model/stylegan/op/fused_bias_act.cpp,./model/stylegan/op/fused_bias_act_kernel.cu,./model/stylegan/op/upfirdn2d.cpp,./model/stylegan/op/conv2d_gradfix.py,./model/encoder/encoders/psp_encoders.py,./model/encoder/encoders/helpers.py,./checkpoint/arcane/vtoonify_s_d.pt,./checkpoint/faceparsing.pth,./checkpoint/encoder.pt,./checkpoint/arcane/exstyle_code.npy,./checkpoint/shape_predictor_68_face_landmarks.dat"])
#         )

        compile_model = torchserve_container

        print(f"repo_url: {repo_url}")
        print(f"repo_url.split('/')[0]: {repo_url.split('/')[0]}")
        print(f"auth_token['user_name']: {auth_token['user_name']}")

        container_publication = await (
            compile_model
            # .with_default_args(args=[
            #     "torchserve",
            #     "--start",
            #     "--model-store=model_store",
            #     "--models",
            #     "vToonify=vToonify.mar",
            #     "--ts-config",
            #     "config.properties"])
            .with_registry_auth(repo_url.split("/")[0], auth_token["user_name"], password)
            .publish(f"{repo_url}:latest")
        )

        # # Destroy ECR
        # ecr_del= await (
        #     ecr_output
        #     .with_env_variable("CACHEBUSTER", str(datetime.now()))
        #     .with_exec(["/bin/bash", "-c", "cd ecr && pulumi stack select dev -c && pulumi up -y"])
        # )

argParser = argparse.ArgumentParser()
argParser.add_argument("--pulumi_token", default=os.environ["PULUMI_ACCESS_TOKEN"], help="PULUMI_ACCESS_TOKEN")
argParser.add_argument("--aws_id", default=os.environ["AWS_ACCESS_KEY_ID"], help="AWS_ACCESS_KEY_ID")
argParser.add_argument("--aws_secret", default=os.environ["AWS_SECRET_ACCESS_KEY"], help="AWS_SECRET_ACCESS_KEY")
argParser.add_argument("--aws_token", default=os.environ["AWS_SESSION_TOKEN"], help="AWS_SESSION_TOKEN")
argParser.add_argument("--stack", default="dev", help="Pulumi stack")

args = argParser.parse_args()
anyio.run(main, args)
