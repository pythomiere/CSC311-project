from pred import predict_all
import pandas as pd

df = pd.read_csv("training_data_202601.csv")
preds = predict_all("training_data_202601.csv")

print(pd.crosstab(df["Painting"], pd.Series(preds, name="Pred")))
print("accuracy =", (df["Painting"].values == preds).mean())