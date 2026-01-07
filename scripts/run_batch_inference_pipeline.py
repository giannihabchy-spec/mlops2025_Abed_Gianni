import os
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor


def main():
    region = os.environ.get("AWS_REGION") or sagemaker.Session().boto_region_name
    role = sagemaker.get_execution_role()
    pipeline_sess = PipelineSession()

    bucket = "mlops2025-abed"

   
    raw_csv_s3 = ParameterString(
        name="RawCsvS3Uri",
        default_value=f"s3://{bucket}/data/batch_input.csv",
    )

   
    model_s3 = ParameterString(
        name="ModelS3Uri",
        default_value=f"s3://{bucket}/models/model.joblib",
    )

    preds_s3 = ParameterString(
        name="PredictionsS3Uri",
        default_value=f"s3://{bucket}/predictions/",
    )

    image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",  
    )

    processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=pipeline_sess,
    )
    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=processor,
        inputs=[
            ProcessingInput(source=raw_csv_s3, destination="/opt/ml/processing/input/raw"),
        ],
        outputs=[
            ProcessingOutput(output_name="clean", source="/opt/ml/processing/output/clean"),
        ],
        code="scripts/preprocess.py",
        job_arguments=[
            "--mode", "inference",
            "--input", "/opt/ml/processing/input/raw/batch_input.csv",
            "--output", "/opt/ml/processing/output/clean/clean.csv",
        ],
    )

    features_step = ProcessingStep(
        name="FeatureEngineering",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs["clean"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/clean",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="features", source="/opt/ml/processing/output/features"),
        ],
        code="scripts/feature_engineering.py",
        job_arguments=[
            "--mode", "inference",
            "--input", "/opt/ml/processing/input/clean/clean.csv",
            "--output", "/opt/ml/processing/output/features/features.csv",
        ],
    )
    inference_step = ProcessingStep(
        name="BatchInference",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=features_step.properties.ProcessingOutputConfig.Outputs["features"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/features",
            ),
            ProcessingInput(
                source=model_s3,
                destination="/opt/ml/processing/input/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source="/opt/ml/processing/output/predictions",
                destination=preds_s3,
            ),
        ],
        code="scripts/batch_inference.py",
        job_arguments=[
            "--features", "/opt/ml/processing/input/features/features.csv",
            "--model", "/opt/ml/processing/input/model/model.joblib",
            "--output", "/opt/ml/processing/output/predictions/predictions.csv",
        ],
    )

    pipeline = Pipeline(
        name="mlops2025-batch-inference-pipeline",
        parameters=[raw_csv_s3, model_s3, preds_s3],
        steps=[preprocess_step, features_step, inference_step],
        sagemaker_session=pipeline_sess,
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("Started batch inference pipeline execution:", execution.arn)


if __name__ == "__main__":
    main()
