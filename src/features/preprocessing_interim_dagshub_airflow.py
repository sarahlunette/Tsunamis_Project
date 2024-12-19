# TODO: comment and unify language
import pandas as pd
import os
from dagshub import get_repo_bucket_client
from io import StringIO, BytesIO
from dagshub import get_repo_bucket_client
repo_name = "Data_Atelier"

s3 = get_repo_bucket_client("sarahlunette/Data_Atelier")

# TODO: change if we change the type of storage DVC onto the local or DVC with DagsHub or PostgreSQL on server or S3 with DVC
url = "https://dagshub.com/sarahlunette/" + repo_name + "/raw/main/data/raw/"


def get_data(table):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(url + table)
    df.columns = df.columns.str.lower()
    # Display the DataFrame
    return df


def create_table(df, name):
    # Convert DataFrame to CSV in memory using StringIO
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)

    # Set the pointer of the StringIO buffer to the beginning
    csv_buffer.seek(0)  # Move to the beginning of the buffer

    # Upload file from the in-memory CSV buffer
    s3.upload_fileobj(
        Fileobj=csv_buffer,  # Use the StringIO buffer directly
        Bucket=repo_name,  # Name of the repo
        Key=name,  # Remote path where to upload the file
    )

    folder_path = '/opt/airflow/data/processed'

    # Create the directory if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df.to_csv("/usr/local/data/processed/" + name)


def write_data():
    tsunamis = get_data("tsunamis.csv")
    gdp = get_data("gdp.csv")
    population = get_data("population.csv")

    columns = [
        "month",
        "day",
        "country",
        "period",
        "latitude",
        "longitude",
        "runup_ht",
        "runup_ht_r",
        "runup_hori",
        "dist_from_",
        "hour",
        "cause_code",
        "event_vali",
        "eq_mag_unk",
        "eq_mag_mb",
        "eq_mag_ms",
        "eq_mag_mw",
        "eq_mag_mfa",
        "eq_magnitu",
        "eq_magni_1",
        "eq_depth",
        "max_event_",
        "ts_mt_ii",
        "ts_intensi",
        "num_runup",
        "num_slides",
        "map_slide_",
        "map_eq_id",
        "gdp_per_capita",
        "houses_damages",
        "human_damages",
    ]

    gdp_per_capita_dict = {
        "afghanistan": 547,
        "bhutan": 3162,
        "channel islands": 37235,
        "cuba": 8616,
        "eritrea": 617,
        "gibraltar": 81000,
        "greenland": 45167,
        "guam": 34529,
        "isle of man": 105375,
        "Not classified": 30000,
        "lebanon": 7556,
        "liechtenstein": 160750,
        "st. martin (french part)": 14000,
        "northern mariana islands": 16200,
        "palau": 14500,
        "korea, dem. people's rep.": 1140,
        "san marino": 69000,
        "south sudan": 268,
        "syrian arab republic": 919,
        "tonga": 5000,
        "venezuela, rb": 6800,
        "british virgin islands": 33333,
        "virgin islands (u.s.)": 38000,
        "taiwan": 35129,
        "venezuela": 15975,
        "turkey": 10674,
        "yemen": 650,
        "egypt": 4295,
        "south korea": 32422,
        "russia": 15262,
        "micronesia": 3714,
    }

    dict_replace = {
        "usa": "united states",
        "usa territory": "united states",
        "myanmar (burma)": "myanmar",
        "uk territory": "united kingdom",
        "east china sea": "china",
        "micronesia, fed. states of": "micronesia",
        "east timor": "indonesia",
        "azores (portugal)": "portugal",
        "uk": "united kingdom",
        "east china sea": "china",
        "cook islands": "france",
        "martinique (french territory)": "france",
    }

    # We acquire the data and then rename the columns
    gdp = gdp[["country name", "2022"]].rename(
        {"country name": "country", "2022": "gdp_per_capita"}, axis=1
    )

    # Then we convert all the values to lowercase
    gdp["country"] = gdp["country"].str.lower()
    tsunamis["country"] = tsunamis["country"].str.lower()

    # We handle outliers and missing values
    gdp = gdp[gdp["country"] != "Not classified"]
    gdp["country"] = gdp["country"].replace(dict_replace)
    # On merge
    tsunamis["country"] = tsunamis["country"].replace(dict_replace)
    data = tsunamis.merge(gdp, on="country", how="left")

    # We do the same thing with population
    population["country"] = population["country"].str.lower()
    population["country"] = population["country"].replace(dict_replace)

    data = data.merge(population, on="country", how="left")

    """ TODO: solve the problem of this line where you have gdp_per_capita as there was the columns to lower"""
    data["GDP_per_capita"] = data["country"].apply(lambda x: gdp_per_capita_dict.get(x))
    data["gdp_per_capita"] = data["gdp_per_capita"].fillna(data["GDP_per_capita"])
    data.drop("GDP_per_capita", axis=1, inplace=True)

    data[["houses_dam", "houses_des"]].fillna(0, inplace=True)
    data["houses_damages"] = data["houses_dam"] + data["houses_des"]

    data[["deaths", "injuries"]].fillna(0, inplace=True)
    data["human_damages"] = data["deaths"] + data["injuries"]

    human_damages = data[
        (~data["human_damages"].isnull()) & (data["human_damages"] != 0)
    ].drop("houses_damages", axis=1)
    houses_damages = data[
        (~data["houses_damages"].isnull()) & (data["houses_damages"] != 0)
    ].drop("human_damages", axis=1)

    houses_damages = houses_damages[columns[:-1]].reset_index(drop=True)
    human_damages = human_damages[columns[:-2] + ["human_damages"]].reset_index(
        drop=True
    )

    human_damages = human_damages.sort_values(by="human_damages")[
        human_damages["human_damages"] < 6000
    ]
    houses_damages = houses_damages.sort_values(by="houses_damages")[
        houses_damages["houses_damages"] < 9000
    ]

    X = pd.get_dummies(human_damages.select_dtypes("object"), dtype=int)
    human_damages = pd.concat([human_damages.drop("country", axis=1), X], axis=1)

    X = pd.get_dummies(houses_damages.select_dtypes("object"), dtype=int)
    houses_damages = pd.concat([houses_damages.drop("country", axis=1), X], axis=1)

    # Get a boto3.client object
    s3 = get_repo_bucket_client("sarahlunette/" + repo_name)

    create_table(human_damages, "human_damages.csv")
    create_table(houses_damages, "houses_damages.csv")
