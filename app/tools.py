import pandas as pd
import glob
import os
from ydata_profiling import ProfileReport
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv


load_dotenv()
api = KaggleApi()
api.authenticate


PATH = "./data"

def search_download_tool(keyword:str) -> str:
    ''' Args: 
    keyword: Dataset keyword for searching in the kaggle 

        Returns:
    path of the downloaded dataset
    '''
    results = api.dataset_list(search=keyword)
    if not results:
        print("No dataset found.")
    else:
        top_dataset = results[0]
        dataset_slug = top_dataset.ref
    api.dataset_download_files(dataset_slug, path=PATH, unzip=True)

    return f"Dataset downloaded to {PATH}"


def eda_report_tool(name:str) -> str:
    ''' Args:
    name: name for the dataset file

    
    Returns A base EDA on the topic in the html file.
    '''
    #get the most recently modified(downloaded) csv file
    csv_files = glob.glob(f"{PATH}/*.csv")
    csv_path = max(csv_files,key=os.path.getmtime) if csv_files else f"{PATH}/"
    df = pd.read_csv(csv_path)
    profile = ProfileReport(df, title=name, explorative=True)
    profile.to_file(f"{name}.html")
    return f"EDA file created."