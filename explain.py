import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load('models/xgb_fair_base.pkl')
X_test = pd.read_csv('dataset/data processed/X_test.csv')
y_test = pd.read_csv('dataset/data processed/y_test.csv')

# Set up explainer
explainer = shap.TreeExplainer(model)
X_sample = X_test.sample(1000, random_state=42)

# Get predictions
y_pred = model.predict(X_sample)

# Find a rejected person
rejected_indices = np.where(y_pred == 1)[0]
i = rejected_indices[0]

print("Explaining prediction for person at index:", i)
print("Features:", X_sample.iloc[i].to_dict())

# Waterfall plot
shap_explanation = explainer(X_sample)
shap.plots.waterfall(shap_explanation[i], show=False)
plt.tight_layout()
plt.savefig('shap_waterfall.png', bbox_inches='tight')
print("Saved shap_waterfall.png")

# Human readable feature names
FEATURE_NAMES = {
    'RevolvingUtilizationOfUnsecuredLines': 'Credit Utilization Rate',
    'age': 'Age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Times 30-59 Days Late',
    'DebtRatio': 'Debt Ratio',
    'MonthlyIncome': 'Monthly Income',
    'NumberOfOpenCreditLinesAndLoans': 'Open Credit Lines',
    'NumberOfTimes90DaysLate': 'Times 90+ Days Late',
    'NumberRealEstateLoansOrLines': 'Real Estate Loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'Times 60-89 Days Late',
    'NumberOfDependents': 'Number of Dependents',
    'TotalPastDue': 'Total Past Due Payments'
}

SUGGESTIONS = {
    'TotalPastDue': 'Reduce your total overdue payments — this is the single biggest factor affecting your score.',
    'RevolvingUtilizationOfUnsecuredLines': 'Try to keep your credit utilization below 30% of your limit.',
    'NumberOfTimes90DaysLate': 'Avoid payments being 90+ days late — this severely impacts your credit profile.',
    'DebtRatio': 'Work on reducing your debt-to-income ratio by paying down existing loans.',
    'MonthlyIncome': 'A higher income or adding a co-applicant can strengthen your application.',
    'NumberOfOpenCreditLinesAndLoans': 'Having too few or too many open credit lines can affect your score.',
    'NumberOfTime60-89DaysPastDueNotWorse': 'Avoid payments being 60-89 days late.',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Even short delays of 30-59 days negatively impact your score.',
    'NumberRealEstateLoansOrLines': 'The number of real estate loans affects your overall debt profile.',
    'NumberOfDependents': 'Number of dependents is a minor factor in your assessment.',
    'age': 'Age is a minor factor — focus on improving your payment history instead.'
}


def get_top3_reasons(shap_vals, feature_names, prediction):
    """
    Returns top 3 reasons for a prediction with human readable names and suggestions.
    prediction: 1 = financial distress (rejected), 0 = approved
    """
    s = pd.Series(shap_vals, index=feature_names)

    if prediction == 1:  # rejected
        top3 = s.nlargest(3)  # most positive = pushed hardest toward rejection
        label = "Top reasons for rejection"
    else:  # approved
        top3 = s.nsmallest(3)  # most negative = pushed hardest toward approval
        label = "Top reasons for approval"

    results = []
    for feature, shap_val in top3.items():
        readable_name = FEATURE_NAMES.get(feature, feature)
        suggestion = SUGGESTIONS.get(
            feature, f'Improving {readable_name} could help.')
        results.append({
            'feature': feature,
            'readable_name': readable_name,
            'shap_value': round(shap_val, 3),
            'suggestion': suggestion
        })

    return results, label


# Test it
shap_vals_single = explainer.shap_values(X_sample)[i]
prediction = y_pred[i]
reasons, label = get_top3_reasons(
    shap_vals_single, X_sample.columns.tolist(), prediction)

print(f"\n{label}:")
for r in reasons:
    print(f"- {r['readable_name']}: {r['shap_value']}")
    print(f"  💡 {r['suggestion']}")
