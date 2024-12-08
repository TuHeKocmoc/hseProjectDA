from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import io

app = FastAPI()

column_names = [
    "CRIM",  # per capita crime rate by town
    "ZN",  # proportion of residential land zoned for lots over 25,000 sq.ft.
    "INDUS",  # proportion of non-retail business acres per town
    "CHAS",
    # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    "NOX",  # nitric oxides concentration (parts per 10 million)
    "RM",  # average number of rooms per dwelling
    "AGE",  # proportion of owner-occupied units built prior to 1940
    "DIS",  # weighted distances to five Boston employment centres
    "RAD",  # index of accessibility to radial highways
    "TAX",  # full-value property-tax rate per $10,000
    "PTRATIO",  # pupil-teacher ratio by town
    "B",  # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    "LSTAT",  # % lower status of the population
    "MEDV"  # Median value of owner-occupied homes in $1000's
]

lines = []
with open('data/boston.txt', 'r') as f:
    all_lines = f.read().strip().split('\n')


data_lines = all_lines[22:]
merged_lines = []
for i in range(0, len(data_lines), 2):
    line = data_lines[i].strip() + " " + data_lines[i + 1].strip()
    merged_lines.append(line)

data = pd.read_csv(io.StringIO('\n'.join(merged_lines)),
                   sep=r'\s+',
                   header=None,
                   names=column_names)

# model for POST-request
class NewHouse(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float
    MEDV: float

# GET for filtering house
@app.get("/houses/")
def get_houses(min_medv: float = 0.0, max_medv: float = 100.0):
    filtered = data[(data['MEDV'] >= min_medv) & (data['MEDV'] <= max_medv)]
    return filtered.to_dict(orient="records")

# POST for adding a new house
@app.post("/houses/")
def add_house(house: NewHouse):
    global data
    new_row = {
        'CRIM': house.CRIM,
        'ZN': house.ZN,
        'INDUS': house.INDUS,
        'CHAS': house.CHAS,
        'NOX': house.NOX,
        'RM': house.RM,
        'AGE': house.AGE,
        'DIS': house.DIS,
        'RAD': house.RAD,
        'TAX': house.TAX,
        'PTRATIO': house.PTRATIO,
        'B': house.B,
        'LSTAT': house.LSTAT,
        'MEDV': house.MEDV
    }
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    return {"message": "House added successfully."}
