# TODO: __init__.py do I need them in folders or above folders for modules
from dagshub import get_repo_bucket_client

# Get a boto3.client object
s3 = get_repo_bucket_client("sarahlunette/Data_Atelier")

# TODO changer le path
path = "/opt/airflow/data/raw/"
# dagspath = 'raw/'


# Upload file
def upload_file(file_path, file_name):
    s3.upload_file(
        Bucket="Data_Atelier",  # name of the repo
        Filename=file_path + file_name,  # local path of file to upload
        Key=file_name,  # remote path where to upload the file
    )

def upload_all():
    upload_file(path, "tsunamis.csv")
    upload_file(path, "gdp.csv")
    upload_file(path, "population.csv")

