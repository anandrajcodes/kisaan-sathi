
from explain_prediction import explain_prediction


sample_input = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 25.0,
    "humidity": 80.0,
    "ph": 6.5,
    "rainfall": 100.0,
}

try:
    explanations = explain_prediction(sample_input)

    print("\nTop feature contributions:\n")

    for item in explanations:
        print(
            f"{item['feature']}: "
            f"value={item['input_value']}, "
            f"contribution={item['shap_value']}, "
            f"direction={item['direction']}"
        )

except Exception as error:
    print(f"Explanation test failed: {error}")
