import os
import sys
import boto3

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from llm_handlers.base_handler import BaseLLMHandler
from llama_index.llms.bedrock import Bedrock
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/test_aws_handler.log")
    
logger = LogManager.get_logger("aws_handler")

class AWSBedrockHandler(BaseLLMHandler):
    """
    Handler for AWS Bedrock LLM service.

    This class extends BaseLLMHandler to provide specific implementation
    for AWS Bedrock LLM service.

    Attributes:
        aws_role_use (bool): Whether to use AWS role for authentication.
        durationSeconds (int): Duration of the AWS session in seconds.
    """
    def __init__(self, model_name, aws_role_use=False, DurationSeconds=900):
        """
        Initialize the AWSBedrockHandler.

        Args:
            model_name (str): The name of the LLM model to use.
            aws_role_use (bool, optional): Whether to use AWS role for authentication. Defaults to False.
            DurationSeconds (int, optional): Duration of the AWS session in seconds. Defaults to 900.
        """
        self.aws_role_use = aws_role_use
        self.durationSeconds = DurationSeconds
        super().__init__(model_name=model_name)
        
        
    def get_external_credentials(self):
        """
        Retrieve AWS credentials for authentication.

        Returns:
            dict: A dictionary containing AWS credentials.
        """
        session_kwargs = {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            }
        
        if self.aws_role_use:
            assumed_role = os.environ.get(
                "AWS_BEDROCK_ROLE", "arn:aws:iam::776026895843:role/policyai-instance")
            logger.info(f"Using AWS user role: {assumed_role}")
        else:
            session_kwargs["profile_name"] = os.environ.get(
                "AWS_PROFILE", "PowerUserAccess-776026895843"
                )
            logger.info(f"Using profile: {session_kwargs['profile_name']}")
        
        session = boto3.Session(**session_kwargs)
        logger.info(f"Create new client using region: {session_kwargs['region_name']}")

        if self.aws_role_use:
            try:
                sts = session.client("sts")
                response = sts.assume_role(
                    RoleArn=str(assumed_role),
                    RoleSessionName="llamaindex-aws-bedrock",
                    DurationSeconds=self.durationSeconds,
                )
                logger.info(f"Credentials obtained for AWS role: {assumed_role}")
            except Exception as e:
                logger.error(e)
                logger.error("Contact admin to give you specific user role. Make sure to include the error message.")

            return {
                "aws_access_key_id": response["Credentials"]["AccessKeyId"],
                "aws_secret_access_key": response["Credentials"]["SecretAccessKey"],
                "aws_session_token": response["Credentials"]["SessionToken"],
                "aws_session_expiration": response["Credentials"]["Expiration"],
                "aws_region_name": session_kwargs["region_name"],
            }

        else:
            try:
                credentials = session.get_credentials().get_frozen_credentials()
                logger.info(f"Frozen credentials obtained for profile: {session_kwargs['profile_name']}")
            except Exception as e:
                logger.error(e)
                logger.error("Configure sso using `aws configure sso`")
        
            return {
                "aws_access_key_id": credentials.access_key,
                "aws_secret_access_key": credentials.secret_key,
                "aws_session_token": credentials.token,
                "aws_session_expiration": None,
                "aws_region_name": session_kwargs["region_name"],
            }

    def get_llm(self)->Bedrock:
        """
        Get an AWS Bedrock LLM client.

        Returns:
            Bedrock: An instance of the Bedrock LLM client.
        """
        llm = Bedrock(
            model=self.model_name,
            aws_access_key_id=self.client_kwargs["aws_access_key_id"],
            aws_secret_access_key=self.client_kwargs["aws_secret_access_key"],
            aws_session_token=self.client_kwargs["aws_session_token"],
            aws_region_name=self.client_kwargs["aws_region_name"],
            context_size=10000
        )
        logger.info(f"AWS cloud (model name: {self.model_name}) is successfully connected.")
        return llm
    
    def generate_response(self, prompt)->str:
        """
        Generate a response using the AWS Bedrock LLM.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response from the LLM.
        """
        response = self.llm.complete(prompt)
        return response.text.strip()


if __name__ == "__main__":
    llm_client = AWSBedrockHandler(model_name="mistral.mistral-7b-instruct-v0:2", aws_role_use=True)