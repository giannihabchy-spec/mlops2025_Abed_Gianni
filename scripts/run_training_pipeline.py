import os
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString

from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn


def main():
    region = os.environ.get("AWS_REGION") or sagemaker.Session().boto_region_name
    role = sagemaker.get_execution_role()
    pipeline_sess = PipelineSession()

    bucket = "mlops2025-abed"

    # -------- Pipeline parameters (override at execution time if needed) --------
    train_features_s3 = ParameterString(
        name="TrainFeaturesS3Uri",
        default_value=f"s3://{bucket}/data/train_features.csv",
    )
    test_features_s3 = ParameterString(
        name="TestFeaturesS3Uri",
        default_value=f"s3://{bucket}/data/test_features.csv",
    )
    model_output_s3 = ParameterString(
        name="ModelOutputS3Uri",
        default_value=f"s3://{bucket}/models/",
    )

    # IMPORTANT: This is the instance that was failing for you (ml.m5.large quota = 0)
    # Default here is ml.t3.medium. If it still fails, try ml.t3.large, ml.m5.xlarge, ml.c5.xlarge.
    train_instance_type = ParameterString(
        name="TrainInstanceType",
        default_value="ml.t3.medium",
    )

    # -------- Real SageMaker Training Job (NOT ProcessingStep) --------
    estimator = SKLearn(
        entry_point="scripts/train_sagemaker.py",
        source_dir=".",              # keep project structure available
        role=role,
        instance_count=1,
        instance_type=train_instance_type,
        framework_version="1.2-1",
        py_version="py3",
        output_path=model_output_s3,
        sagemaker_session=pipeline_sess,
    )

    train_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(s3_data=train_features_s3, content_type="text/csv"),
            "test": TrainingInput(s3_data=test_features_s3, content_type="text/csv"),
        },
    )

    pipeline = Pipeline(
        name="mlops2025-training-pipeline",
        parameters=[train_features_s3, test_features_s3, model_output_s3, train_instance_type],
        steps=[train_step],
        sagemaker_session=pipeline_sess,
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("Started training pipeline execution:", execution.arn)


if __name__ == "__main__":
    main()


