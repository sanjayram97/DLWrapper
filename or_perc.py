from utils.model import Perceptron
import pandas as pd
from utils.all_utils import prepare_data


df = pd.DataFrame()

df["x1"] = [1, 1, 0, 0]
df["x2"] = [0, 1, 0, 1]
df["y"] = [1, 1, 0, 1]

x, y = prepare_data(df)

model_or = Perceptron(lr=0.2, epochs=5)
model_or.fit(x, y)

print("Total Loss: ", model_or.total_loss())