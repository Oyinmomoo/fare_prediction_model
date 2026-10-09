#import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#Correlation Analysis
file_path="Data-Travel-Prices_Re-sit.xls"
data=pd.read_excel(r"C:\Users\adeni\OneDrive\Desktop\Origin\Airfare Price prediction model\Source\Airfare_source_data.xlsx",sheet_name="data")
print(data.shape)
print (data.columns.tolist())

numeric_vars = [
    "COUPON",
    "NEW",
    "HI",
    "S_INCOME",
    "E_INCOME",
    "S_POP",
    "E_POP",
    "DISTANCE",
    "PAX",
    "FARE"]

fare_correlations = (
data[numeric_vars]
    .corr()["FARE"]
    .drop("FARE")
    .sort_values(key=abs, ascending=False)
)

print(fare_correlations.round(3))

#Correlation plot with trend line 
import matplotlib.pyplot as plt
import numpy as np

predictors = [v for v in numeric_vars if v != "FARE"]

for variable in predictors:
    
    plot_data = data[[variable, "FARE"]].dropna()
    x = plot_data[variable]
    y = plot_data["FARE"]
    
    correlation = x.corr(y)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    plt.figure(figsize=(7, 5))
    plt.scatter(
        x,
        y,
        alpha=0.6
    )

    plt.plot(
        x,
        y_pred,
        linewidth=2
    )
    plt.xlabel(variable)
    plt.ylabel("FARE ($)")
    plt.title(f"FARE vs {variable}")
    
    plt.text(
        0.05,
        0.95,
        f"Correlation = {correlation:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )
    
    plt.tight_layout()
    plt.show()

#Average fare using Categorical predictors
categorical_vars =[
    "VACATION"
    "SW"
    "SLOT"
    "GATE"
]
for variable in categorical_vars:
    percentages = (
        data["VACATION"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
)
print("VACATION")
print(percentages)

for variable in categorical_vars:
    percentages = (
        data["SW"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
)
print("SW")
print(percentages)

for variable in categorical_vars:
    percentages = (
        data["SLOT"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
)
print("SLOT")
print(percentages)

for variable in categorical_vars:
    percentages = (
        data["GATE"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
)
print("GATE")
print(percentages)


for variable in categorical_vars:
    print("Average FARE by VACATION")
    print(
        data.groupby("VACATION")["FARE"]
        .mean()
        .round(2)
)

for variable in categorical_vars:
    print("Average FARE by SW")
    print(
        data.groupby("SW")["FARE"]
        .mean()
        .round(2)
)

for variable in categorical_vars:
    print("Average FARE by SLOT")
    print(
        data.groupby("SLOT")["FARE"]
        .mean()
        .round(2)
)

for variable in categorical_vars:
    print("Average FARE by GATE")
    print(
        data.groupby("GATE")["FARE"]
        .mean()
        .round(2)
)

#Convert categorical variables
data["VACATION_D"] = (data["VACATION"] == "Yes").astype(int)
data["SW_D"] = (data["SW"] == "Yes").astype(int)
data["SLOT_D"] = (data["SLOT"] == "Free").astype(int)
data["GATE_D"] = (data["GATE"] == "Free").astype(int)

predictors = [
    "COUPON",
    "NEW",
    "VACATION_D",
    "SW_D",
    "HI",
    "S_INCOME",
    "E_INCOME",
    "S_POP",
    "E_POP",
    "SLOT_D",
    "GATE_D",
    "DISTANCE",
    "PAX"
]

target = "FARE"

print(data[["VACATION", "VACATION_D",
            "SW", "SW_D",
            "SLOT", "SLOT_D",
            "GATE", "GATE_D"]].head())

#Partition data into training and validation
from sklearn.model_selection import train_test_split


X = data[predictors]
y = data["FARE"]

print("X shape:", X.shape)
print("y shape:", y.shape)

a_train, a_valid, b_train, b_valid = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=123456
)

print("Training data:", a_train.shape)
print("Validation data:",a_valid.shape)

#Using stepwise regression to select top predictors
import statsmodels.api as sm

selected_stepwise = ["DISTANCE", "SW_D", "VACATION_D", "HI", "GATE_D",
    "SLOT_D", "PAX", "E_INCOME", "S_POP", "E_POP", "S_INCOME"]

while True:
    current_aic = (
        sm.OLS(b_train, sm.add_constant(a_train[selected_stepwise])).fit().aic
        if selected_stepwise else float("inf")
    )

    results = []
    for col in a_train.columns:
        if col not in selected_stepwise:
            cols = selected_stepwise + [col]
            model = sm.OLS(
                b_train, 
                sm.add_constant(a_train[cols])
            ).fit().aic
            results.append((current_aic, col))

    best_aic, best_col = min(results)

    if best_aic < current_aic:
        selected.append(best_col)
    else:
        break

print("Stepwise selected predictors:")
print(selected_stepwise)

#Stepwise regression summary
stepwise_model = sm.OLS(
    b_train,
    sm.add_constant(a_train[selected_stepwise])
).fit()

print(stepwise_model.summary())

#Data info for second model
from itertools import combinations

results = [stepwise_model.summary]

all_predictors = list(a_train.columns)

print("Number of predictors:", len(all_predictors))
print("Predictors:", all_predictors)
print("Total models to evaluate:", 2**len(all_predictors) -1)

#Using exhaustive search to select top predictors
results = []
for r in range(1, len(all_predictors) + 1):
    
    for subset in combinations(all_predictors, r):
        model= sm.OLS(
            b_train,
            sm.add_constant(a_train[list(subset)])
        ).fit()

        results.append({
            "predictors": subset,
            "Number_of_predictors": r,
            "AIC": model.aic,
            "BIC": model.bic,
            "Adjusted_R2": model.rsquared_adj,
                "Cp": np.nan
        })

exhaustive_results = pd.DataFrame(results)

print("Number of models evaluated:", len(exhaustive_results))

print("Top 10 models by AIC:")
best_exhaustive = (
    exhaustive_results
    .sort_values("AIC")
    .head(10)
)
print(exhaustive_results)  

#Best exhaustive search model predictors
best_exhaustive = exhaustive_results.sort_values("AIC").iloc[0]

print("Best exhaustive model:")
print(best_exhaustive)

print ("Complete predictor list:")
print(best_exhaustive["predictors"])

#Compare models using RMSE and MAE
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

best_predictors = list(best_exhaustive["predictors"])

stepwise_model = sm.OLS(
    b_train,
    sm.add_constant(a_train[all_predictors])
).fit()

exhaustive_model = sm.OLS(
    b_train,
    sm.add_constant(a_train[best_predictors])
).fit()

stepwise_pred = stepwise_model.predict(
    sm.add_constant(
        a_valid[all_predictors],
        has_constant="add"
    )
)

exhaustive_pred = exhaustive_model.predict(
    sm.add_constant(
        a_valid[best_predictors],
        has_constant="add"
    )
)

stepwise_rmse = np.sqrt(
    mean_squared_error(
        b_valid,
        stepwise_pred
    )
)

exhaustive_rmse = np.sqrt(
    mean_squared_error(
        b_valid,
        exhaustive_pred
    )
)

stepwise_mae = mean_absolute_error(
    b_valid,
    stepwise_pred
)

exhaustive_mae = mean_absolute_error(
    b_valid,
    exhaustive_pred
)

comparison = pd.DataFrame({
    "Model": ["Stepwise", "Exhaustive"],
    "RMSE": [stepwise_rmse, exhaustive_rmse],
    "MAE": [stepwise_mae, exhaustive_mae]
})

print(comparison)

print("Difference in RMSE: $", round(stepwise_rmse - exhaustive_rmse))
print("Difference in MAE: $", round(stepwise_mae - exhaustive_mae))

#Lift Chart 
import matplotlib.pyplot as plt
import numpy as np

lift_data =pd.DataFrame({
    "Actual_FARE": b_valid.values,
    "Stepwise_Predicted": stepwise_pred.values,
    "Exhaustive_Predicted": exhaustive_pred.values
})

lift_data = lift_data.sort_values("Actual_FARE", ascending=False).reset_index(drop=True)

lift_data["Cumulative_Actual"] = lift_data["Actual_FARE"].cumsum()
lift_data["Cumulative_Stepwise"] = lift_data["Stepwise_Predicted"].cumsum()
lift_data["Cumulative_Exhaustive"] = lift_data["Exhaustive_Predicted"].cumsum()

lift_data["Cumulative_Actual_%"] = ( 
    lift_data["Cumulative_Actual"] / lift_data["Actual_FARE"].sum() * 100
)

lift_data["Cumulative_Stepwise_%"] = (
    lift_data["Cumulative_Stepwise"] / lift_data["Stepwise_Predicted"].sum() * 100
)

lift_data["Cumulative_Exhaustive_%"] = (
    lift_data["Cumulative_Exhaustive"] / lift_data["Cumulative_Exhaustive"].sum() * 100
)

x = np.arange(1, len(lift_data) + 1) / len(lift_data) * 100

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    lift_data["Cumulative_Actual_%"],
    label="Actual FARE"
)

plt.plot(
    x,
    lift_data["Cumulative_Stepwise_%"],
    label="Stepwise Model"
)

plt.plot(
    x,
    lift_data["Cumulative_Exhaustive_%"],
    label="Exhaustive Model"
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    label="Random/Baseline"
)

plt.xlabel("Percentage of validation observations")
plt.ylabel("Cumulative percentage of FARE")
plt.title("Lift Chart: Stepwise vs Exhaustive Regression")
plt.legend()
plt.grid(True)
plt.show()

#Using new characteristics to predict fare price
new_route = pd.DataFrame({
    "COUPON": [1.202],
    "NEW": [3],
    "VACATION_D": [0],  
    "SW_D": [0],         
    "HI": [4442.141],
    "S_INCOME": [28760],
    "E_INCOME": [27664],
    "S_POP": [4557004],
    "E_POP": [3195503],
    "SLOT_D": [1],       
    "GATE_D": [1],       
    "DISTANCE": [1976],
    "PAX": [12782]
})

predicted_fare = exhaustive_model.predict(
    sm.add_constant(new_route[best_predictors], has_constant="add")
)

print("Predicted average fare: $", round(predicted_fare.iloc[0], 2))

#Fare prediction without Southwest
sw_coefficient = exhaustive_model.params["SW_D"]

print("SW_D coefficient:", round(sw_coefficient, 2))
print("Predicted fare reduction: $", round(abs(sw_coefficient), 2))

#Using only pre-launch predictors
prelaunch_predictors = [
    "COUPON",
    "NEW",
    "VACATION_D",
    "SW_D",
    "S_INCOME",
    "E_INCOME",
    "S_POP",
    "E_POP",
    "SLOT_D",
    "GATE_D",
    "DISTANCE"
]

test_results = []

for r in range(1, len(prelaunch_predictors) + 1):
    
    for subset in combinations(prelaunch_predictors, r):
        test_results.append(subset)

print("Number of combinations stored:", len(test_results))
print("First combination:", test_results[0])
print("Last combination:", test_results[-1])

#Using exhaustive model with pre-launch predictors
prelaunch_results = []

for subset in test_results:
    X_subset = sm.add_constant(
        a_train[list(subset)]
)
    model = sm.OLS(
        b_train,
        X_subset
    ).fit()

    prelaunch_results.append({
        "Predictors": subset,
        "Number_of_predictors": len(subset),
        "AIC": model.aic,
        "BIC": model.bic,
        "Adjusted_R2": model.rsquared_adj
})

prelaunch_results = pd.DataFrame(prelaunch_results)

print("Number of models evaluated:", len(prelaunch_results))
print(type(prelaunch_results))
print(prelaunch_results[:3])

#pre-launch summary

prelaunch_best = prelaunch_results.sort_values("AIC").iloc[0]

print("Best pre-launch model:")
print("Predictors:", prelaunch_best["Predictors"])
print("Number of predictors:", prelaunch_best["Number_of_predictors"])
print("AIC:", round(prelaunch_best["AIC"], 3))
print("BIC:", round(prelaunch_best["BIC"], 3))
print("Adjusted R-squared:", round(prelaunch_best["Adjusted_R2"], 3))

#Using new characteristics to predict fare for pre-launch model
prelaunch_predictors = list(prelaunch_best["Predictors"])

X_prelaunch = sm.add_constant(a_train[prelaunch_predictors])
prelaunch_model = sm.OLS(b_train, X_prelaunch).fit()

route = pd.DataFrame({
    "VACATION_D": [0],      
    "SW_D": [0],            
    "E_INCOME": [27664],
    "SLOT_D": [1],          
    "GATE_D": [1],         
    "DISTANCE": [1976]
})

route = sm.add_constant(route, has_constant="add")

predicted_fare = prelaunch_model.predict(route).iloc[0]

print("Selected predictors:", prelaunch_predictors)
print("Predicted average fare: $", round(predicted_fare, 2))

#Compare pre-launch and exhaustive search model
prelaunch_model = sm.OLS(
    b_train, sm.add_constant(a_train[prelaunch_predictors])
).fit()

prelaunch_predictions = prelaunch_model.predict(
    sm.add_constant(a_valid[prelaunch_predictors], has_constant="add")
)

prelaunch_rmse = np.sqrt(
    mean_squared_error(b_valid, prelaunch_predictions)
)
prelaunch_mae = mean_absolute_error(
    b_valid,
    prelaunch_predictions
)

print("Pre-launch model RMSE: $", round(prelaunch_rmse, 2))
print("Exhaustive Model RMSE: $", round(exhaustive_rmse, 2))
print("Difference in RMSE: $", round(prelaunch_rmse - exhaustive_rmse))

print("Pre-launch model MAE: $", round(prelaunch_mae, 2))
print("Exhaustive Model MAE: $", round(exhaustive_mae, 2))
print("Difference in MAE: $", round(prelaunch_mae - exhaustive_mae))